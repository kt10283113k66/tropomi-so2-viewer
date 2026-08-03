from __future__ import annotations

import io
import csv
import gc
import shutil
import json
import math
import os
import tempfile
import time as time_module
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import rasterio
import pandas as pd
import requests
import streamlit as st
import xarray as xr
from scipy.ndimage import label as connected_component_label

from core import (
    MSM_P_BASE_URL,
    add_peak_fitting_diagnostics,
    calculate_quasi_steady_model,
    choose_optimal_candidate,
    data_filter,
    find_annulus_maximum,
    float_so2_evalscript,
    get_access_token,
    get_secret,
    local_xy_m,
    radius_bbox,
    read_float_tiff,
    request_process,
)

APP_DIR = Path(__file__).resolve().parent
VOLCANO_FILE = APP_DIR / "volcanoes.json"
MSM_CACHE_DIR = APP_DIR / ".cache" / "msm_p"
RESULTS_DIR = APP_DIR / "results"
LOG_DIR = APP_DIR / "logs"
PLUME_THRESHOLD_MOL_M2 = 0.001
MIN_CONNECTED_PLUME_PIXELS = 3
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_VOLCANOES = {
    "volcanoes": [
        {
            "name": "阿蘇山",
            "crater": "中岳第一火口",
            "latitude": 32.8847282,
            "longitude": 131.0848191,
        },
        {
            "name": "桜島",
            "crater": "南岳山頂火口",
            "latitude": 31.5769,
            "longitude": 130.6583,
        },
        {
            "name": "十勝岳",
            "crater": "62-2火口",
            "latitude": 43.423234,
            "longitude": 142.675452,
        },
    ]
}

CSV_COLUMNS = [
    "日付",
    "補正放出率_t_day",
    "TROPOMIピークカラム濃度_mol_m2",
    "TROPOMIピーク流下距離_km",
    "TROPOMIピーク流向_deg",
    "TROPOMIモデル流向差_deg",
    "採用モデル時刻_JST",
    "採用モデル気圧面_hPa",
    "採用モデル火口風速_m_s",
    "採用モデル火口風向_deg",
    "モデル分解能_南北_km",
    "モデル分解能_東西_km",
    "ステータス",
    "エラー内容",
]


def load_volcanoes() -> list[dict]:
    if not VOLCANO_FILE.exists():
        save_volcanoes(DEFAULT_VOLCANOES["volcanoes"])
    try:
        payload = json.loads(VOLCANO_FILE.read_text(encoding="utf-8"))
        items = payload.get("volcanoes", [])
    except Exception:
        items = DEFAULT_VOLCANOES["volcanoes"]
    valid = []
    for item in items:
        try:
            valid.append(
                {
                    "name": str(item["name"]).strip(),
                    "crater": str(item["crater"]).strip(),
                    "latitude": float(item["latitude"]),
                    "longitude": float(item["longitude"]),
                }
            )
        except Exception:
            continue
    return valid


def save_volcanoes(items: list[dict]) -> None:
    VOLCANO_FILE.write_text(
        json.dumps({"volcanoes": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def volcano_label(item: dict) -> str:
    return f"{item['name']}（{item['crater']}）"


def _haversine_km(lat1, lon1, lat2, lon2):
    earth_radius = 6371.0088
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1r)
        * np.cos(lat2r)
        * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * earth_radius * np.arcsin(
        np.sqrt(np.clip(a, 0.0, 1.0))
    )


def _bearing_deg(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)

    x = math.sin(dlon) * math.cos(lat2r)
    y = (
        math.cos(lat1r) * math.sin(lat2r)
        - math.sin(lat1r)
        * math.cos(lat2r)
        * math.cos(dlon)
    )
    return (
        math.degrees(math.atan2(x, y)) + 360.0
    ) % 360.0


def find_annulus_maximum(
    array: np.ndarray,
    transform,
    crater_lat: float,
    crater_lon: float,
    min_distance_km: float = 10.0,
    max_distance_km: float = 30.0,
    threshold_mol_m2: float = PLUME_THRESHOLD_MOL_M2,
    min_connected_pixels: int = MIN_CONNECTED_PLUME_PIXELS,
):
    """
    10～30 km環状帯で、閾値以上の8近傍連結成分を検出する。

    閾値以上のピクセルが3個以上、辺または角で隣接する成分だけを
    有効なプルーム候補とし、その成分内の最大値をピークとして返す。
    """
    rows, cols = np.indices(array.shape)
    xs, ys = rasterio.transform.xy(
        transform,
        rows,
        cols,
        offset="center",
    )

    longitudes = np.asarray(
        xs,
        dtype=float,
    ).reshape(array.shape)
    latitudes = np.asarray(
        ys,
        dtype=float,
    ).reshape(array.shape)

    distances = _haversine_km(
        crater_lat,
        crater_lon,
        latitudes,
        longitudes,
    )

    plume_mask = (
        np.isfinite(array)
        & (array >= float(threshold_mol_m2))
        & (distances >= float(min_distance_km))
        & (distances <= float(max_distance_km))
    )

    if not np.any(plume_mask):
        return None

    # 上下左右と斜めを含む8近傍。
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, component_count = connected_component_label(
        plume_mask,
        structure=structure,
    )

    candidates = []
    for component_id in range(1, component_count + 1):
        component_mask = labels == component_id
        pixel_count = int(
            np.count_nonzero(component_mask)
        )
        if pixel_count < int(min_connected_pixels):
            continue

        component_values = np.where(
            component_mask,
            array,
            -np.inf,
        )
        flat_index = int(np.argmax(component_values))
        row, col = np.unravel_index(
            flat_index,
            component_values.shape,
        )

        candidates.append(
            {
                "row": int(row),
                "col": int(col),
                "pixel_count": pixel_count,
                "peak_value": float(array[row, col]),
            }
        )

    if not candidates:
        return None

    selected = max(
        candidates,
        key=lambda candidate: candidate["peak_value"],
    )

    row = selected["row"]
    col = selected["col"]
    peak_latitude = float(latitudes[row, col])
    peak_longitude = float(longitudes[row, col])

    return {
        "value": float(array[row, col]),
        "latitude": peak_latitude,
        "longitude": peak_longitude,
        "distance_km": float(distances[row, col]),
        "bearing_deg": _bearing_deg(
            crater_lat,
            crater_lon,
            peak_latitude,
            peak_longitude,
        ),
        "connected_pixel_count": int(
            selected["pixel_count"]
        ),
        "threshold_mol_m2": float(
            threshold_mol_m2
        ),
        "minimum_connected_pixels": int(
            min_connected_pixels
        ),
    }


def apply_emission_qa(
    model_result: dict,
    tropomi_peak_value: float,
    correction_factor: float,
) -> dict:
    """
    放出率推定のQA条件を適用する。

    条件:
    - TROPOMIピークカラム濃度 >= 0.001 mol/m²
    - TROPOMIとモデルのピーク流向差 <= 45°
    """
    peak_value = float(tropomi_peak_value)
    direction_difference = float(
        model_result["peak_direction_difference_deg"]
    )

    qa_reasons = []
    if peak_value < PLUME_THRESHOLD_MOL_M2:
        qa_reasons.append(
            "TROPOMIピークカラム濃度が0.001 mol/m²未満"
        )
    if direction_difference > 45.0:
        qa_reasons.append(
            "TROPOMIとモデルの流向差が45°を超過"
        )

    qa_pass = len(qa_reasons) == 0
    fitted_flux = model_result.get(
        "fitted_emission_rate_t_day"
    )

    corrected_flux = None
    if qa_pass:
        if fitted_flux is None:
            raise RuntimeError(
                "推定放出率が取得できないため、補正放出率を算出できません。"
            )
        corrected_flux = float(fitted_flux) * float(
            correction_factor
        )

    model_result["emission_qa_pass"] = qa_pass
    model_result["emission_qa_reasons"] = qa_reasons
    model_result["corrected_emission_rate_t_day"] = (
        corrected_flux
    )
    return model_result


def date_sequence(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def clear_msm_cache() -> tuple[int, int]:
    before = directory_size_bytes(MSM_CACHE_DIR)
    removed = 0
    if MSM_CACHE_DIR.exists():
        for item in sorted(MSM_CACHE_DIR.rglob("*"), reverse=True):
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink(missing_ok=True)
                    removed += 1
                elif item.is_dir():
                    item.rmdir()
            except OSError:
                continue
    MSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return before, removed


def msm_dates_for_analysis(target_date: date, jst_hours: list[int]) -> set[date]:
    dates = set()
    for jst_hour in jst_hours:
        jst_datetime = datetime.combine(target_date, time(hour=int(jst_hour)))
        dates.add((jst_datetime - timedelta(hours=9)).date())
    return dates


def delete_msm_files(target_dates: set[date]) -> int:
    removed = 0
    for target_date in target_dates:
        path = msm_file_path(target_date)
        for candidate in (path, path.with_suffix(".part")):
            try:
                if candidate.exists():
                    candidate.unlink()
                    removed += 1
            except OSError:
                pass
        try:
            path.parent.rmdir()
        except OSError:
            pass
    return removed


def safe_filename(text: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in text)
    return safe.strip("_") or "volcano"


def append_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as file_object:
        writer = csv.DictWriter(file_object, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
        file_object.flush()


def append_log(path: Path, target_date: date, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_object:
        file_object.write(
            f"{datetime.now():%Y-%m-%d %H:%M:%S}	"
            f"{target_date.isoformat()}	"
            f"{row.get('ステータス', '')}	"
            f"{row.get('エラー内容', '')}\n"
        )
        file_object.flush()


def msm_file_path(target_date: date) -> Path:
    return (
        MSM_CACHE_DIR
        / f"{target_date.year:04d}"
        / f"{target_date.month:02d}{target_date.day:02d}.nc"
    )


def ensure_msm_file(target_date: date) -> Path:
    target = msm_file_path(target_date)
    if target.exists() and target.stat().st_size > 1_000_000:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part")
    url = (
        f"{MSM_P_BASE_URL}/{target_date.year:04d}/"
        f"{target_date.month:02d}{target_date.day:02d}.nc"
    )
    try:
        with requests.get(
            url,
            headers={"User-Agent": "TROPOMI-SO2-Batch/1.0"},
            stream=True,
            timeout=(30, 600),
        ) as response:
            if response.status_code == 404:
                raise RuntimeError(
                    f"MSM-Pデータが見つかりません：{target_date.isoformat()}"
                )
            response.raise_for_status()
            with temporary.open("wb") as file_object:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file_object.write(chunk)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def extract_msm_field(
    target_date: date,
    utc_hour: int,
    pressure_level: int,
    crater_latitude: float,
    crater_longitude: float,
):
    path = ensure_msm_file(target_date)
    requested_time = np.datetime64(
        datetime.combine(target_date, time(hour=int(utc_hour)))
    )

    try:
        with xr.open_dataset(path) as ds:
            for coordinate in ("time", "p", "lat", "lon"):
                if coordinate not in ds.coords:
                    raise RuntimeError(f"MSM-Pに座標{coordinate}がありません。")
            if "u" not in ds.data_vars or "v" not in ds.data_vars:
                raise RuntimeError("MSM-Pにuまたはvがありません。")

            selected = ds.sel(
                time=requested_time,
                p=float(pressure_level),
                method="nearest",
            )
            lat_min = crater_latitude - 1.2
            lat_max = crater_latitude + 1.2
            lon_min = crater_longitude - 1.5
            lon_max = crater_longitude + 1.5
            selected = selected.where(
                (selected["lat"] >= lat_min) & (selected["lat"] <= lat_max),
                drop=True,
            )
            selected = selected.where(
                (selected["lon"] >= lon_min) & (selected["lon"] <= lon_max),
                drop=True,
            )
            if selected.sizes.get("lat", 0) == 0 or selected.sizes.get("lon", 0) == 0:
                raise RuntimeError("火口周辺のMSM-P格子を抽出できません。")

            latitudes = np.asarray(selected["lat"].values, dtype=float)
            longitudes = np.asarray(selected["lon"].values, dtype=float)
            u = np.asarray(selected["u"].values, dtype=float).squeeze()
            v = np.asarray(selected["v"].values, dtype=float).squeeze()
            actual_time = np.asarray(selected["time"]).astype("datetime64[s]").item()

            geopotential_height = None
            for candidate in ("z", "gh", "geopotential_height"):
                if candidate in selected.data_vars:
                    z_point = selected[candidate].sel(
                        lat=crater_latitude,
                        lon=crater_longitude,
                        method="nearest",
                    )
                    geopotential_height = float(np.asarray(z_point).squeeze())
                    break
    except Exception as error:
        raise RuntimeError(f"MSM-Pの読み込みに失敗しました：{error}") from error

    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    x_grid, y_grid = local_xy_m(
        lat_grid,
        lon_grid,
        crater_latitude,
        crater_longitude,
    )
    valid = np.isfinite(u) & np.isfinite(v)
    if not np.any(valid):
        raise RuntimeError("MSM-P風データが欠測です。")

    if isinstance(actual_time, datetime):
        actual_datetime = actual_time
    else:
        actual_datetime = datetime.fromisoformat(str(actual_time))

    return {
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
        "x_grid_m": x_grid,
        "y_grid_m": y_grid,
        "u_ms": u,
        "v_ms": v,
        "valid": valid,
        "analysis_time": actual_datetime.strftime("%Y-%m-%d %H:%M UTC"),
        "utc_hour": int(actual_datetime.hour),
        "pressure_level": int(pressure_level),
        "source_path": str(path),
        "geopotential_height_m": geopotential_height,
    }


def process_one_date(
    target_date: date,
    volcano: dict,
    token: str,
    min_qa: int,
    display_radius_km: float,
    image_size: int,
    selected_jst_hours: list[int],
    pressure_levels: list[int],
    assumed_flux_t_day: float,
    correction_factor: float,
    maximum_axis_km: float,
    lateral_half_width_km: float,
    model_grid_size: int,
    model_resolution_ns_km: float,
    model_resolution_ew_km: float,
) -> dict:
    start_time = time_module.time()
    result = {column: "" for column in CSV_COLUMNS}
    result["日付"] = target_date.isoformat()
    result["モデル分解能_南北_km"] = model_resolution_ns_km
    result["モデル分解能_東西_km"] = model_resolution_ew_km

    crater_lat = float(volcano["latitude"])
    crater_lon = float(volcano["longitude"])
    bbox = radius_bbox(crater_lat, crater_lon, display_radius_km)

    try:
        so2_tiff = request_process(
            token=token,
            bbox=bbox,
            target_date=target_date,
            size=image_size,
            min_qa=min_qa,
            evalscript=float_so2_evalscript(),
            mime_type="image/tiff",
        )
        so2_array, so2_transform = read_float_tiff(so2_tiff)
        maximum = find_annulus_maximum(
            array=so2_array,
            transform=so2_transform,
            crater_lat=crater_lat,
            crater_lon=crater_lon,
            min_distance_km=10.0,
            max_distance_km=30.0,
            threshold_mol_m2=PLUME_THRESHOLD_MOL_M2,
            min_connected_pixels=MIN_CONNECTED_PLUME_PIXELS,
        )
        if maximum is None:
            result["ステータス"] = "NO_COHERENT_PLUME"
            result["エラー内容"] = (
                "10～30 km内で、0.001 mol/m²以上のピクセルが"
                "8近傍で3個以上連結するプルームを検出できません。"
            )
            return result

        candidates = []
        candidate_errors = []
        for jst_hour in sorted(selected_jst_hours):
            jst_datetime = datetime.combine(
                target_date,
                time(hour=int(jst_hour)),
            )
            utc_datetime = jst_datetime - timedelta(hours=9)
            for pressure_level in sorted(pressure_levels, reverse=True):
                try:
                    field = extract_msm_field(
                        utc_datetime.date(),
                        utc_datetime.hour,
                        pressure_level,
                        crater_lat,
                        crater_lon,
                    )
                    model_result = calculate_quasi_steady_model(
                        msm_field=field,
                        crater_latitude=crater_lat,
                        crater_longitude=crater_lon,
                        bbox=bbox,
                        output_size=model_grid_size,
                        emission_rate_t_day=assumed_flux_t_day,
                        maximum_distance_km=maximum_axis_km,
                        lateral_half_width_km=lateral_half_width_km,
                        tropomi_peak_distance_km=maximum["distance_km"],
                        peak_distance_half_width_km=5.0,
                        target_ns_km=model_resolution_ns_km,
                        target_ew_km=model_resolution_ew_km,
                    )
                    model_result = add_peak_fitting_diagnostics(
                        model_result,
                        maximum,
                        assumed_flux_t_day,
                    )
                    # QA判定は最適モデルを選択した後に適用する。
                    actual_jst_hour = (int(field["utc_hour"]) + 9) % 24
                    candidates.append(
                        {
                            "jst_hour": int(actual_jst_hour),
                            "pressure_level": int(pressure_level),
                            "model_result": model_result,
                            "msm_field": field,
                        }
                    )
                except Exception as error:
                    candidate_errors.append(
                        f"{jst_hour:02d}JST/{pressure_level}hPa: {error}"
                    )

        if not candidates:
            result["ステータス"] = "MODEL_ERROR"
            result["エラー内容"] = " | ".join(candidate_errors)[:2000]
            return result

        optimal, _reason = choose_optimal_candidate(candidates)
        model = optimal["model_result"]

        model = apply_emission_qa(
            model_result=model,
            tropomi_peak_value=float(maximum["value"]),
            correction_factor=correction_factor,
        )

        corrected_flux_output = ""
        status = "SUCCESS"
        error_message = ""

        if model["emission_qa_pass"]:
            corrected_flux_output = round(
                float(model["corrected_emission_rate_t_day"]),
                3,
            )
        else:
            status = "QA_FAIL"
            error_message = " / ".join(
                model["emission_qa_reasons"]
            )

        result.update(
            {
                "補正放出率_t_day": corrected_flux_output,
                "TROPOMIピークカラム濃度_mol_m2": float(maximum["value"]),
                "TROPOMIピーク流下距離_km": round(
                    float(maximum["distance_km"]), 3
                ),
                "TROPOMIピーク流向_deg": round(
                    float(maximum["bearing_deg"]), 3
                ),
                "TROPOMIモデル流向差_deg": round(
                    float(model["peak_direction_difference_deg"]), 3
                ),
                "採用モデル時刻_JST": int(optimal["jst_hour"]),
                "採用モデル気圧面_hPa": int(optimal["pressure_level"]),
                "採用モデル火口風速_m_s": round(
                    float(model["crater_speed_ms"]), 3
                ),
                "採用モデル火口風向_deg": round(
                    float(model["crater_wind_from_deg"]), 3
                ),
                "ステータス": status,
                "エラー内容": error_message,
            }
        )
        return result
    except requests.Timeout:
        result["ステータス"] = "API_TIMEOUT"
        result["エラー内容"] = "TROPOMI APIがタイムアウトしました。"
        return result
    except Exception as error:
        result["ステータス"] = "ERROR"
        result["エラー内容"] = str(error)[:2000]
        return result


st.set_page_config(
    page_title="TROPOMI SO₂ Batch Processor",
    page_icon="🌋",
    layout="wide",
)
st.title("TROPOMI SO₂ 期間一括解析")
st.caption(
    "TROPOMIピークとMSM-P準定常モデルから、日別の補正放出率を一括計算します。"
)
st.info(
    "プルーム検出条件：火口から10～30 kmで、"
    "0.001 mol/m²以上のピクセルが8近傍で3個以上連結すること。"
    "この条件を満たさない日はNO_COHERENT_PLUMEとして記録し、"
    "MSM-P取得とモデル計算を行いません。"
    "放出率は、さらにTROPOMIとモデルの流向差が45°以下の場合だけ"
    "算出します。"
)

analysis_tab, volcano_tab = st.tabs(["期間一括解析", "火山管理"])

with volcano_tab:
    st.subheader("火山管理")
    volcanoes = load_volcanoes()
    if volcanoes:
        st.dataframe(
            pd.DataFrame(volcanoes).rename(
                columns={
                    "name": "name",
                    "crater": "crater",
                    "latitude": "緯度",
                    "longitude": "経度",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    add_tab, edit_tab, delete_tab = st.tabs(["追加", "編集", "削除"])
    with add_tab:
        with st.form("add_volcano_form"):
            new_name = st.text_input("name")
            new_crater = st.text_input("crater")
            new_latitude = st.number_input(
                "緯度", min_value=-90.0, max_value=90.0, format="%.7f"
            )
            new_longitude = st.number_input(
                "経度", min_value=-180.0, max_value=180.0, format="%.7f"
            )
            add_submitted = st.form_submit_button("追加して保存")
        if add_submitted:
            if not new_name.strip() or not new_crater.strip():
                st.error("nameとcraterを入力してください。")
            else:
                volcanoes.append(
                    {
                        "name": new_name.strip(),
                        "crater": new_crater.strip(),
                        "latitude": float(new_latitude),
                        "longitude": float(new_longitude),
                    }
                )
                save_volcanoes(volcanoes)
                st.success("追加しました。ページを再読み込みすると一覧へ反映されます。")

    with edit_tab:
        if volcanoes:
            selected_edit_label = st.selectbox(
                "編集する火山",
                [volcano_label(v) for v in volcanoes],
                key="edit_volcano",
            )
            edit_index = [volcano_label(v) for v in volcanoes].index(
                selected_edit_label
            )
            current = volcanoes[edit_index]
            with st.form("edit_volcano_form"):
                edit_name = st.text_input("name", value=current["name"])
                edit_crater = st.text_input("crater", value=current["crater"])
                edit_latitude = st.number_input(
                    "緯度",
                    min_value=-90.0,
                    max_value=90.0,
                    value=float(current["latitude"]),
                    format="%.7f",
                )
                edit_longitude = st.number_input(
                    "経度",
                    min_value=-180.0,
                    max_value=180.0,
                    value=float(current["longitude"]),
                    format="%.7f",
                )
                edit_submitted = st.form_submit_button("変更を保存")
            if edit_submitted:
                volcanoes[edit_index] = {
                    "name": edit_name.strip(),
                    "crater": edit_crater.strip(),
                    "latitude": float(edit_latitude),
                    "longitude": float(edit_longitude),
                }
                save_volcanoes(volcanoes)
                st.success("変更を保存しました。")

    with delete_tab:
        if volcanoes:
            delete_label = st.selectbox(
                "削除する火山",
                [volcano_label(v) for v in volcanoes],
                key="delete_volcano",
            )
            confirm_delete = st.checkbox("削除を確認しました")
            if st.button("削除", disabled=not confirm_delete):
                volcanoes = [
                    v for v in volcanoes if volcano_label(v) != delete_label
                ]
                save_volcanoes(volcanoes)
                st.success("削除しました。")

with analysis_tab:
    volcanoes = load_volcanoes()
    if not volcanoes:
        st.error("火山が登録されていません。火山管理タブで追加してください。")
        st.stop()

    with st.sidebar:
        st.header("一括解析設定")
        selected_label = st.selectbox(
            "火山",
            [volcano_label(v) for v in volcanoes],
        )
        volcano = next(v for v in volcanoes if volcano_label(v) == selected_label)

        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input(
                "開始日", value=date.today() - timedelta(days=7)
            )
        with col_end:
            end_date = st.date_input(
                "終了日", value=date.today() - timedelta(days=1)
            )

        min_qa = st.slider("最小QA値（%）", 0, 100, 50, 5)
        display_radius_km = st.selectbox(
            "解析範囲半径", [40, 50, 60, 80, 100], index=2
        )
        image_size = st.select_slider(
            "TROPOMI取得格子", [256, 384, 512, 768], value=512
        )

        selected_jst_hours = st.multiselect(
            "モデル時刻（JST）",
            options=list(range(0, 24, 3)),
            default=[12, 15],
            format_func=lambda value: f"{value:02d}:00",
        )
        pressure_levels = st.multiselect(
            "気圧面（hPa）",
            [1000, 950, 900, 850, 800, 700, 500],
            default=[900, 850, 800],
        )
        pattern_count = len(selected_jst_hours) * len(pressure_levels)
        st.caption(f"計算パターン数：{pattern_count} / 6")

        assumed_flux_t_day = st.number_input(
            "仮定SO₂放出率（t/day）",
            min_value=1.0,
            max_value=100000.0,
            value=1000.0,
            step=100.0,
        )
        correction_factor = st.number_input(
            "補正倍率",
            min_value=0.01,
            max_value=100.0,
            value=3.0,
            step=0.1,
        )
        maximum_axis_km = st.selectbox(
            "主軸計算距離（km）", [30, 40, 50], index=2
        )
        lateral_half_width_km = st.selectbox(
            "主軸横幅（±km）", [10, 15, 20, 30], index=2
        )
        model_grid_size = st.select_slider(
            "モデル内部計算格子", [256, 384, 512], value=384
        )

        st.markdown("##### モデル平均化分解能")
        res1, res2 = st.columns(2)
        with res1:
            model_resolution_ns_km = st.number_input(
                "南北（km）", 0.5, 30.0, 3.5, 0.1
            )
        with res2:
            model_resolution_ew_km = st.number_input(
                "東西（km）", 0.5, 30.0, 7.0, 0.1
            )

        st.markdown("##### 保存・ディスク設定")
        keep_temporary_files = st.checkbox(
            "MSM-P一時ファイルを保存する",
            value=False,
            help=(
                "OFF推奨。OFFでは各日の解析終了後にMSM-P NetCDFを削除します。"
            ),
        )
        resume_existing = st.checkbox(
            "既存CSVから再開する",
            value=True,
            help="同じ火山・期間のCSVがある場合、記録済みの日付をスキップします。",
        )

        current_cache_size = directory_size_bytes(MSM_CACHE_DIR)
        st.caption(f"現在のMSM-Pキャッシュ：{human_size(current_cache_size)}")
        if st.button("MSM-Pキャッシュを今すぐ削除", use_container_width=True):
            released, removed_count = clear_msm_cache()
            st.success(
                f"{removed_count}ファイルを削除し、"
                f"約{human_size(released)}を解放しました。"
            )

        valid_settings = (
            start_date <= end_date
            and 1 <= pattern_count <= 6
        )
        run_batch = st.button(
            "一括解析を開始",
            type="primary",
            use_container_width=True,
            disabled=not valid_settings,
        )

    st.write(
        f"対象：**{volcano_label(volcano)}**　"
        f"緯度 {volcano['latitude']:.7f} / 経度 {volcano['longitude']:.7f}"
    )

    client_id = get_secret("CDSE_CLIENT_ID")
    client_secret = get_secret("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        st.warning(
            "`.streamlit/secrets.toml`にCDSE_CLIENT_IDと"
            "CDSE_CLIENT_SECRETを設定してください。"
        )

    if run_batch:
        if not client_id or not client_secret:
            st.stop()

        run_name = (
            f"{safe_filename(volcano_label(volcano))}_"
            f"{start_date:%Y%m%d}_{end_date:%Y%m%d}"
        )
        result_path = RESULTS_DIR / f"{run_name}.csv"
        log_path = LOG_DIR / f"{run_name}.log"

        existing_rows = []
        completed_dates = set()
        if resume_existing and result_path.exists():
            try:
                existing_df = pd.read_csv(result_path, dtype=str)
                existing_rows = existing_df.to_dict("records")
                completed_dates = set(existing_df.get("日付", pd.Series(dtype=str)).dropna())
            except Exception as error:
                st.warning(f"既存CSVを読み込めなかったため新規解析します：{error}")

        dates = list(date_sequence(start_date, end_date))
        pending_dates = [d for d in dates if d.isoformat() not in completed_dates]

        if not pending_dates:
            st.info("指定期間はすべて既存CSVに記録済みです。")
            result_df = pd.DataFrame(existing_rows, columns=CSV_COLUMNS)
            st.session_state["batch_result_df"] = result_df
            st.session_state["batch_result_path"] = str(result_path)
        else:
            progress = st.progress(0.0, text="解析を開始します…")
            status_placeholder = st.empty()
            table_placeholder = st.empty()
            rows = list(existing_rows)

            for index, target_date in enumerate(pending_dates, start=1):
                token = get_access_token(client_id, client_secret)
                status_placeholder.info(
                    f"{target_date.isoformat()} を解析中（{index}/{len(pending_dates)}）"
                )

                row = process_one_date(
                    target_date=target_date,
                    volcano=volcano,
                    token=token,
                    min_qa=min_qa,
                    display_radius_km=float(display_radius_km),
                    image_size=int(image_size),
                    selected_jst_hours=list(selected_jst_hours),
                    pressure_levels=list(pressure_levels),
                    assumed_flux_t_day=float(assumed_flux_t_day),
                    correction_factor=float(correction_factor),
                    maximum_axis_km=float(maximum_axis_km),
                    lateral_half_width_km=float(lateral_half_width_km),
                    model_grid_size=int(model_grid_size),
                    model_resolution_ns_km=float(model_resolution_ns_km),
                    model_resolution_ew_km=float(model_resolution_ew_km),
                )

                # 1日ごとに永続化するため、途中停止しても結果が残る。
                append_csv_row(result_path, row)
                append_log(log_path, target_date, row)
                rows.append(row)

                if not keep_temporary_files:
                    delete_msm_files(
                        msm_dates_for_analysis(
                            target_date,
                            list(selected_jst_hours),
                        )
                    )

                # 大きな配列・バイト列を日ごとに解放。
                del row
                gc.collect()

                progress.progress(
                    index / len(pending_dates),
                    text=f"{index}/{len(pending_dates)}日 完了",
                )
                table_placeholder.dataframe(
                    pd.DataFrame(rows, columns=CSV_COLUMNS).tail(20),
                    use_container_width=True,
                    hide_index=True,
                )

            # 念のため、未保存モードでは残存キャッシュも削除する。
            if not keep_temporary_files:
                clear_msm_cache()

            result_df = pd.read_csv(result_path)
            st.session_state["batch_result_df"] = result_df
            st.session_state["batch_result_path"] = str(result_path)
            st.session_state["batch_log_path"] = str(log_path)
            status_placeholder.success(
                "一括解析が完了しました。"
                f" 結果は {result_path.name} に保存されています。"
            )

    if "batch_result_df" in st.session_state:
        result_df = st.session_state["batch_result_df"]
        st.subheader("解析結果")
        summary1, summary2, summary3, summary4 = st.columns(4)
        summary1.metric("対象日数", len(result_df))
        summary2.metric(
            "放出率算出日数",
            int((result_df["ステータス"] == "SUCCESS").sum()),
        )
        summary3.metric(
            "QA不適合日数",
            int((result_df["ステータス"] == "QA_FAIL").sum()),
        )
        summary4.metric(
            "エラー・欠測日数",
            int(
                (
                    ~result_df["ステータス"].isin(
                        ["SUCCESS", "QA_FAIL"]
                    )
                ).sum()
            ),
        )
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        result_path_value = st.session_state.get("batch_result_path")
        if result_path_value and Path(result_path_value).exists():
            result_path = Path(result_path_value)
            st.caption(f"自動保存先：{result_path}")
            csv_bytes = result_path.read_bytes()
            csv_name = result_path.name
        else:
            csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
            csv_name = (
                f"tropomi_so2_batch_{start_date:%Y%m%d}_"
                f"{end_date:%Y%m%d}.csv"
            )

        st.download_button(
            "CSVを一括ダウンロード",
            data=csv_bytes,
            file_name=csv_name,
            mime="text/csv",
            use_container_width=True,
        )

        log_path_value = st.session_state.get("batch_log_path")
        if log_path_value and Path(log_path_value).exists():
            log_path = Path(log_path_value)
            st.download_button(
                "解析ログをダウンロード",
                data=log_path.read_bytes(),
                file_name=log_path.name,
                mime="text/plain",
                use_container_width=True,
            )
