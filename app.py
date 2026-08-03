import io
import json
import math
import os
import tempfile
from pathlib import Path
from datetime import date, datetime, time, timedelta, timezone

import time as time_module
import folium
import numpy as np
import rasterio
import requests
import streamlit as st
import xarray as xr
from scipy.spatial import cKDTree
from scipy.ndimage import label as connected_component_label
from branca.element import Element
from folium.plugins import Fullscreen, MousePosition
from PIL import Image, ImageDraw
from rasterio.io import MemoryFile
from streamlit_folium import st_folium


TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
MSM_P_BASE_URL = "http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/netcdf/MSM-P"
SO2_MOLAR_MASS_KG_MOL = 0.064066
PLUME_THRESHOLD_MOL_M2 = 0.001
MIN_CONNECTED_PLUME_PIXELS = 3
MAX_DIRECTION_DIFFERENCE_DEG = 45.0

VOLCANO_FILE = Path(__file__).resolve().parent / "volcanoes.json"


@st.cache_data(show_spinner=False)
def load_volcanoes() -> list[dict]:
    """volcanoes.jsonから火山・火口位置を読み込む。"""
    if not VOLCANO_FILE.exists():
        raise RuntimeError(
            f"火山情報ファイルが見つかりません：{VOLCANO_FILE}"
        )

    try:
        with VOLCANO_FILE.open("r", encoding="utf-8") as file_object:
            payload = json.load(file_object)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"volcanoes.jsonの形式が正しくありません：{error}"
        ) from error
    except OSError as error:
        raise RuntimeError(
            f"volcanoes.jsonを読み込めません：{error}"
        ) from error

    volcanoes = payload.get("volcanoes")
    if not isinstance(volcanoes, list) or not volcanoes:
        raise RuntimeError(
            "volcanoes.jsonのvolcanoesは、1件以上の配列にしてください。"
        )

    required = {"name", "crater", "latitude", "longitude"}
    cleaned = []

    for index, item in enumerate(volcanoes, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(
                f"volcanoes.jsonの{index}件目がオブジェクトではありません。"
            )

        missing = required - set(item)
        if missing:
            raise RuntimeError(
                f"volcanoes.jsonの{index}件目に不足項目があります："
                + ", ".join(sorted(missing))
            )

        name = str(item["name"]).strip()
        crater = str(item["crater"]).strip()

        try:
            latitude = float(item["latitude"])
            longitude = float(item["longitude"])
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"volcanoes.jsonの{index}件目の緯度経度が数値ではありません。"
            ) from error

        if not name:
            raise RuntimeError(
                f"volcanoes.jsonの{index}件目のnameが空です。"
            )
        if not crater:
            raise RuntimeError(
                f"volcanoes.jsonの{index}件目のcraterが空です。"
            )
        if not (-90.0 <= latitude <= 90.0):
            raise RuntimeError(
                f"{name}の緯度が範囲外です：{latitude}"
            )
        if not (-180.0 <= longitude <= 180.0):
            raise RuntimeError(
                f"{name}の経度が範囲外です：{longitude}"
            )

        cleaned.append(
            {
                "name": name,
                "crater": crater,
                "latitude": latitude,
                "longitude": longitude,
            }
        )

    return cleaned


def volcano_display_name(volcano: dict) -> str:
    return f"{volcano['name']}（{volcano['crater']}）"


# 指定された固定階級
SO2_BOUNDS = [
    0.0004, 0.0010, 0.0015, 0.0020, 0.0030, 0.0040, 0.0050,
    0.0060, 0.0070, 0.0080, 0.0090, 0.0100, 0.0200
]

# 添付画像に近い青紫→水色→緑→黄→赤→桃→紫
SO2_COLORS = [
    "#7767E8",
    "#62A9EE",
    "#58D8EC",
    "#55EDC2",
    "#69ED78",
    "#A3ED55",
    "#F5F227",
    "#FFAC49",
    "#FA7775",
    "#ED69B3",
    "#DF5BE9",
    "#B656E2",
    "#87008F",
]

SO2_LABELS = [
    "0.0004–0.001",
    "0.001–0.0015",
    "0.0015–0.002",
    "0.002–0.003",
    "0.003–0.004",
    "0.004–0.005",
    "0.005–0.006",
    "0.006–0.007",
    "0.007–0.008",
    "0.008–0.009",
    "0.009–0.01",
    "0.01–0.02",
    "0.02–",
]



def require_app_password():
    """
    Streamlit CloudのSecretsにAPP_PASSWORDが設定されている場合だけ、
    簡易パスワード認証を有効にする。
    """
    try:
        configured_password = str(
            st.secrets.get("APP_PASSWORD", "")
        ).strip()
    except Exception:
        configured_password = str(
            os.getenv("APP_PASSWORD", "")
        ).strip()

    # 未設定なら認証なしで公開
    if not configured_password:
        return

    if st.session_state.get("app_authenticated", False):
        return

    st.title("TROPOMI SO₂ Viewer")
    st.info("このアプリを利用するにはパスワードが必要です。")

    entered_password = st.text_input(
        "パスワード",
        type="password",
        key="public_app_password",
    )

    if st.button(
        "ログイン",
        type="primary",
        use_container_width=True,
    ):
        if entered_password == configured_password:
            st.session_state["app_authenticated"] = True
            st.rerun()
        else:
            st.error("パスワードが正しくありません。")

    st.stop()


def get_secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.getenv(name, "")).strip()


def get_access_token(
    client_id: str | None = None,
    client_secret: str | None = None,
    force_refresh: bool = False,
) -> str:
    """CDSEアクセストークンを取得し、有効期限を管理する。"""

    now = time_module.time()

    if not force_refresh:
        token = st.session_state.get("cdse_access_token")
        expires_at = st.session_state.get("cdse_token_expires_at", 0.0)

        # 有効期限の60秒前までは保存済みトークンを使用する
        if token and now < float(expires_at) - 60.0:
            return str(token)

    client_id = (client_id or get_secret("CDSE_CLIENT_ID")).strip()
    client_secret = (client_secret or get_secret("CDSE_CLIENT_SECRET")).strip()

    if not client_id or not client_secret:
        raise RuntimeError(
            "CDSE_CLIENT_IDまたはCDSE_CLIENT_SECRETが未設定です。"
        )

    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )

    if not response.ok:
        raise RuntimeError(
            f"CDSE認証に失敗しました（HTTP {response.status_code}）。\n"
            f"{response.text[:1000]}"
        )

    token_data = response.json()
    access_token = token_data.get("access_token")
    if not access_token:
        raise RuntimeError("CDSE認証応答にaccess_tokenが含まれていません。")

    expires_in = int(token_data.get("expires_in", 3600))
    st.session_state["cdse_access_token"] = access_token
    st.session_state["cdse_token_expires_at"] = now + expires_in

    return str(access_token)


def utc_range(target_date: date) -> tuple[str, str]:
    start = datetime.combine(target_date, time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return (
        start.isoformat().replace("+00:00", "Z"),
        end.isoformat().replace("+00:00", "Z"),
    )


def data_filter(target_date: date) -> dict:
    start, end = utc_range(target_date)
    return {
        "timeRange": {"from": start, "to": end},
        "mosaickingOrder": "mostRecent",
    }


def hex_rgb_fraction(hex_color: str) -> tuple[float, float, float]:
    return tuple(
        int(hex_color[i:i + 2], 16) / 255.0
        for i in (1, 3, 5)
    )


def fixed_so2_evalscript() -> str:
    statements = []
    for i, upper in enumerate(SO2_BOUNDS[1:]):
        r, g, b = hex_rgb_fraction(SO2_COLORS[i])
        statements.append(
            f"if (v < {upper:.10g}) "
            f"return [{r:.8f}, {g:.8f}, {b:.8f}, sample.dataMask];"
        )

    r, g, b = hex_rgb_fraction(SO2_COLORS[-1])

    return f"""
//VERSION=3
function setup() {{
  return {{
    input: ["SO2", "dataMask"],
    output: {{ bands: 4, sampleType: "AUTO" }}
  }};
}}

function evaluatePixel(sample) {{
  const v = sample.SO2;

  if (sample.dataMask === 0 || !isFinite(v) || v < {SO2_BOUNDS[0]:.10g}) {{
    return [0, 0, 0, 0];
  }}

  {' '.join(statements)}

  return [{r:.8f}, {g:.8f}, {b:.8f}, sample.dataMask];
}}
"""


def float_so2_evalscript() -> str:
    return """
//VERSION=3
function setup() {
  return {
    input: ["SO2", "dataMask"],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  if (sample.dataMask === 1 && isFinite(sample.SO2)) {
    return [sample.SO2];
  }
  return [-9999.0];
}
"""


def cloud_evalscript() -> str:
    return """
//VERSION=3
function setup() {
  return {
    input: ["CLOUD_FRACTION", "dataMask"],
    output: { bands: 4, sampleType: "AUTO" }
  };
}

function evaluatePixel(sample) {
  if (sample.dataMask === 0 || !isFinite(sample.CLOUD_FRACTION)) {
    return [0, 0, 0, 0];
  }

  const c = Math.max(0.0, Math.min(1.0, sample.CLOUD_FRACTION));
  return [1.0, 1.0, 1.0, c * sample.dataMask];
}
"""


def float_cloud_evalscript() -> str:
    return """
//VERSION=3
function setup() {
  return {
    input: ["CLOUD_FRACTION", "dataMask"],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  if (sample.dataMask === 1 && isFinite(sample.CLOUD_FRACTION)) {
    return [sample.CLOUD_FRACTION];
  }
  return [-9999.0];
}
"""


def request_process(
    token: str,
    bbox: list[float],
    target_date: date,
    size: int,
    min_qa: int,
    evalscript: str,
    mime_type: str,
) -> bytes:
    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                },
            },
            "data": [{
                "type": "sentinel-5p-l2",
                "dataFilter": data_filter(target_date),
                "processing": {
                    "minQa": min_qa,
                    "upsampling": "NEAREST",
                    "downsampling": "NEAREST",
                },
            }],
        },
        "output": {
            "width": size,
            "height": size,
            "responses": [{
                "identifier": "default",
                "format": {"type": mime_type},
            }],
        },
        "evalscript": evalscript,
    }

    def send_request(access_token: str):
        return requests.post(
            PROCESS_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Accept": mime_type,
            },
            json=payload,
            timeout=120,
        )

    response = send_request(token)

    # アクセストークン期限切れの場合は、新しいトークンを取得して1回だけ再送する
    if response.status_code == 401:
        refreshed_token = get_access_token(force_refresh=True)
        response = send_request(refreshed_token)

    if not response.ok:
        raise RuntimeError(
            f"データ取得に失敗しました（HTTP {response.status_code}）。\n"
            f"{response.text[:1000]}"
        )

    return response.content



def request_process_rectangular(
    token: str,
    bbox: list[float],
    target_date: date,
    width: int,
    height: int,
    min_qa: int,
    evalscript: str,
    mime_type: str,
) -> bytes:
    """縦横別の格子数を指定してSentinel Hub Process APIから取得する。"""
    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                },
            },
            "data": [{
                "type": "sentinel-5p-l2",
                "dataFilter": data_filter(target_date),
                "processing": {
                    "minQa": min_qa,
                    "upsampling": "NEAREST",
                    "downsampling": "NEAREST",
                },
            }],
        },
        "output": {
            "width": int(width),
            "height": int(height),
            "responses": [{
                "identifier": "default",
                "format": {"type": mime_type},
            }],
        },
        "evalscript": evalscript,
    }

    def send_request(access_token: str):
        return requests.post(
            PROCESS_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Accept": mime_type,
            },
            json=payload,
            timeout=120,
        )

    response = send_request(token)
    if response.status_code == 401:
        response = send_request(
            get_access_token(force_refresh=True)
        )
    if not response.ok:
        raise RuntimeError(
            f"連結判定用データ取得に失敗しました（HTTP {response.status_code}）。\n"
            f"{response.text[:1000]}"
        )
    return response.content


def plume_detection_grid_size(
    bbox: list[float],
    center_latitude: float,
    target_ns_km: float = 3.5,
    target_ew_km: float = 7.0,
) -> tuple[int, int]:
    """TROPOMI相当の物理分解能から連結判定用の幅・高さを求める。"""
    north_south_km = abs(bbox[3] - bbox[1]) * 111.32
    east_west_km = (
        abs(bbox[2] - bbox[0])
        * 111.32
        * max(math.cos(math.radians(center_latitude)), 0.05)
    )
    height = max(3, int(math.ceil(north_south_km / target_ns_km)))
    width = max(3, int(math.ceil(east_west_km / target_ew_km)))
    return width, height


def radius_bbox(
    latitude: float,
    longitude: float,
    radius_km: float,
) -> list[float]:
    lat_delta = radius_km / 111.32
    lon_delta = radius_km / (
        111.32 * max(math.cos(math.radians(latitude)), 0.05)
    )
    return [
        longitude - lon_delta,
        latitude - lat_delta,
        longitude + lon_delta,
        latitude + lat_delta,
    ]


def haversine_km(lat1, lon1, lat2, lon2):
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


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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

    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def read_float_tiff(tiff_bytes: bytes):
    with MemoryFile(tiff_bytes) as memory_file:
        with memory_file.open() as src:
            array = src.read(1).astype(float)
            transform = src.transform
            nodata = src.nodata

    array[~np.isfinite(array)] = np.nan
    array[array <= -9000.0] = np.nan

    if nodata is not None and np.isfinite(nodata):
        array[array == nodata] = np.nan

    return array, transform


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
    10～30 km環状帯で、閾値以上の8近傍連結成分を抽出する。

    3画素以上が辺または角で隣接する成分だけをプルーム候補とし、
    候補成分内で最も高いSO₂カラム濃度をピークとして返す。
    """
    rows, cols = np.indices(array.shape)

    xs, ys = rasterio.transform.xy(
        transform,
        rows,
        cols,
        offset="center",
    )
    longitudes = np.asarray(xs, dtype=float).reshape(array.shape)
    latitudes = np.asarray(ys, dtype=float).reshape(array.shape)

    distances = haversine_km(
        crater_lat,
        crater_lon,
        latitudes,
        longitudes,
    )

    plume_mask = (
        np.isfinite(array)
        & (array >= float(threshold_mol_m2))
        & (distances >= min_distance_km)
        & (distances <= max_distance_km)
    )

    if not np.any(plume_mask):
        return None

    # 8近傍（上下左右＋斜め）で連結成分を判定する。
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, component_count = connected_component_label(
        plume_mask,
        structure=structure,
    )

    valid_components = []
    for component_id in range(1, component_count + 1):
        component_mask = labels == component_id
        pixel_count = int(np.count_nonzero(component_mask))

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

        valid_components.append(
            {
                "row": int(row),
                "col": int(col),
                "pixel_count": pixel_count,
                "peak_value": float(array[row, col]),
            }
        )

    if not valid_components:
        return None

    selected = max(
        valid_components,
        key=lambda item: item["peak_value"],
    )
    row = selected["row"]
    col = selected["col"]

    max_lat = float(latitudes[row, col])
    max_lon = float(longitudes[row, col])

    return {
        "value": float(array[row, col]),
        "latitude": max_lat,
        "longitude": max_lon,
        "distance_km": float(distances[row, col]),
        "bearing_deg": bearing_deg(
            crater_lat,
            crater_lon,
            max_lat,
            max_lon,
        ),
        "connected_pixel_count": int(
            selected["pixel_count"]
        ),
        "threshold_mol_m2": float(threshold_mol_m2),
        "minimum_connected_pixels": int(
            min_connected_pixels
        ),
    }



def value_at_click(array, transform, latitude: float, longitude: float):
    row, col = rasterio.transform.rowcol(
        transform,
        longitude,
        latitude,
    )

    if (
        row < 0
        or col < 0
        or row >= array.shape[0]
        or col >= array.shape[1]
    ):
        return None

    value = array[row, col]
    if not np.isfinite(value):
        return None

    return float(value)


def angle_difference_deg(a, b):
    return np.abs((np.asarray(a) - b + 180.0) % 360.0 - 180.0)


def coordinate_grids(array: np.ndarray, transform):
    rows, cols = np.indices(array.shape)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    longitudes = np.asarray(xs, dtype=float).reshape(array.shape)
    latitudes = np.asarray(ys, dtype=float).reshape(array.shape)
    return latitudes, longitudes


def pixel_areas_m2(array: np.ndarray, transform):
    """緯度経度格子の各セル面積を球面近似で計算する。"""
    latitudes, _ = coordinate_grids(array, transform)
    dlon = abs(float(transform.a))
    dlat = abs(float(transform.e))
    earth_radius = 6_371_008.8
    lat_north = np.radians(latitudes + dlat / 2.0)
    lat_south = np.radians(latitudes - dlat / 2.0)
    return (
        earth_radius ** 2
        * math.radians(dlon)
        * np.abs(np.sin(lat_north) - np.sin(lat_south))
    )



def local_xy_m(latitude, longitude, origin_latitude, origin_longitude):
    """緯度経度を火口基準の局所直交座標[m]へ近似変換する。"""
    earth_radius = 6_371_008.8
    x = (
        np.radians(np.asarray(longitude) - origin_longitude)
        * earth_radius
        * math.cos(math.radians(origin_latitude))
    )
    y = np.radians(np.asarray(latitude) - origin_latitude) * earth_radius
    return x, y


def xy_to_latlon(x_m, y_m, origin_latitude, origin_longitude):
    earth_radius = 6_371_008.8
    latitude = origin_latitude + np.degrees(np.asarray(y_m) / earth_radius)
    longitude = origin_longitude + np.degrees(
        np.asarray(x_m)
        / (earth_radius * math.cos(math.radians(origin_latitude)))
    )
    return latitude, longitude


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_msm_p_field(
    target_date: date,
    utc_hour: int,
    pressure_level: int,
    crater_latitude: float,
    crater_longitude: float,
):
    """MSM-P日別NetCDFから指定時刻・気圧面の2次元風場を取得する。"""
    year = f"{target_date.year:04d}"
    filename = f"{target_date.month:02d}{target_date.day:02d}.nc"
    url = f"{MSM_P_BASE_URL}/{year}/{filename}"
    headers = {"User-Agent": "TROPOMI-SO2-Viewer/1.0"}

    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / filename

        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(30, 300),
            ) as response:
                if response.status_code == 404:
                    raise RuntimeError(
                        f"MSM-Pデータが見つかりません（{target_date.isoformat()}）。"
                        "公開前または欠測の可能性があります。"
                    )
                response.raise_for_status()
                with target.open("wb") as file_object:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file_object.write(chunk)
        except requests.Timeout as error:
            raise RuntimeError("MSM-Pデータの取得がタイムアウトしました。") from error
        except requests.RequestException as error:
            raise RuntimeError(f"MSM-Pデータの取得に失敗しました：{error}") from error

        requested_time = np.datetime64(
            datetime.combine(target_date, time(hour=int(utc_hour)))
        )

        try:
            with xr.open_dataset(target) as ds:
                for coordinate in ("time", "p", "lat", "lon"):
                    if coordinate not in ds.coords:
                        raise RuntimeError(
                            f"MSM-Pファイルに座標 {coordinate} がありません。"
                        )
                if "u" not in ds.data_vars or "v" not in ds.data_vars:
                    raise RuntimeError("MSM-Pファイルにuまたはvが含まれていません。")

                available_levels = np.asarray(ds["p"].values, dtype=float)
                if not np.any(np.isclose(available_levels, pressure_level)):
                    raise RuntimeError(
                        f"{pressure_level} hPaはMSM-Pに収録されていません。"
                    )

                selected = ds.sel(
                    time=requested_time,
                    p=float(pressure_level),
                    method="nearest",
                )

                # 主軸50 km＋補間半径10 kmを十分含むよう火口周辺を切り出す。
                # MSM-Pのlat座標はファイルによって昇順・降順が異なるため、
                # sliceではなく条件式で抽出する。
                lat_min = crater_latitude - 1.2
                lat_max = crater_latitude + 1.2
                lon_min = crater_longitude - 1.5
                lon_max = crater_longitude + 1.5

                selected = selected.where(
                    (selected["lat"] >= lat_min)
                    & (selected["lat"] <= lat_max),
                    drop=True,
                )
                selected = selected.where(
                    (selected["lon"] >= lon_min)
                    & (selected["lon"] <= lon_max),
                    drop=True,
                )

                if selected.sizes.get("lat", 0) == 0:
                    raise RuntimeError(
                        "火口周辺のMSM-P緯度格子を抽出できませんでした。"
                    )
                if selected.sizes.get("lon", 0) == 0:
                    raise RuntimeError(
                        "火口周辺のMSM-P経度格子を抽出できませんでした。"
                    )

                latitudes = np.asarray(selected["lat"].values, dtype=float)
                longitudes = np.asarray(selected["lon"].values, dtype=float)
                u = np.asarray(selected["u"].values, dtype=float).squeeze()
                v = np.asarray(selected["v"].values, dtype=float).squeeze()

                if u.ndim != 2 or v.ndim != 2:
                    raise RuntimeError("MSM-Pのu・vが2次元格子として読み込めません。")

                actual_time = np.asarray(selected["time"]).astype("datetime64[s]").item()

                geopotential_height = None
                for candidate in ("z", "gh", "geopotential_height"):
                    if candidate in selected.data_vars:
                        z_data = selected[candidate]
                        if (
                            z_data.sizes.get("lat", 0) > 0
                            and z_data.sizes.get("lon", 0) > 0
                        ):
                            z_point = z_data.sel(
                                lat=crater_latitude,
                                lon=crater_longitude,
                                method="nearest",
                            )
                            geopotential_height = float(
                                np.asarray(z_point).squeeze()
                            )
                        break
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"MSM-P NetCDFの読み込みに失敗しました：{error}"
            ) from error

    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    x_grid, y_grid = local_xy_m(
        lat_grid,
        lon_grid,
        crater_latitude,
        crater_longitude,
    )

    valid = np.isfinite(u) & np.isfinite(v)
    if not np.any(valid):
        raise RuntimeError("MSM-Pの風データが欠測です。")

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
        "source_url": url,
        "geopotential_height_m": geopotential_height,
    }


def interpolate_idw_wind(msm_field, x_m: float, y_m: float, radius_m: float = 10_000.0):
    """論文の式(6)(7)に従い、半径10 km内を距離二乗逆数で内挿する。"""
    valid = msm_field["valid"]
    gx = msm_field["x_grid_m"][valid]
    gy = msm_field["y_grid_m"][valid]
    gu = msm_field["u_ms"][valid]
    gv = msm_field["v_ms"][valid]

    distance = np.hypot(gx - x_m, gy - y_m)
    in_radius = distance <= radius_m

    if np.any(distance < 1.0):
        idx = int(np.argmin(distance))
        return float(gu[idx]), float(gv[idx])

    if not np.any(in_radius):
        idx = int(np.argmin(distance))
        return float(gu[idx]), float(gv[idx])

    d = distance[in_radius]
    weights = 1.0 / np.maximum(d, 1.0) ** 2
    u = float(np.sum(weights * gu[in_radius]) / np.sum(weights))
    v = float(np.sum(weights * gv[in_radius]) / np.sum(weights))
    return u, v


def calculate_main_axis(
    msm_field,
    maximum_distance_km: float,
    nominal_step_m: float = 100.0,
):
    """式(1)～(7)に基づき、MSM-P固定時刻の風場から主軸を計算する。"""
    u0, v0 = interpolate_idw_wind(msm_field, 0.0, 0.0)
    speed0 = math.hypot(u0, v0)
    if speed0 < 0.2:
        raise RuntimeError("火口上空の風速が小さすぎるため主軸を計算できません。")

    delta_t = nominal_step_m / speed0
    maximum_distance_m = maximum_distance_km * 1000.0

    xs = [0.0]
    ys = [0.0]
    distances = [0.0]
    speeds = [speed0]

    for _ in range(int(maximum_distance_m / 20.0) + 5000):
        u, v = interpolate_idw_wind(msm_field, xs[-1], ys[-1])
        speed = math.hypot(u, v)
        if speed < 0.1:
            break

        new_x = xs[-1] + u * delta_t
        new_y = ys[-1] + v * delta_t
        segment = math.hypot(new_x - xs[-1], new_y - ys[-1])

        xs.append(new_x)
        ys.append(new_y)
        distances.append(distances[-1] + segment)
        speeds.append(speed)

        if distances[-1] >= maximum_distance_m:
            break

    if len(xs) < 3:
        raise RuntimeError("ガス主軸を十分な長さまで計算できませんでした。")

    latitudes, longitudes = xy_to_latlon(
        np.asarray(xs),
        np.asarray(ys),
        float(msm_field["lat_grid"].mean()),
        float(msm_field["lon_grid"].mean()),
    )

    # 上記平均値ではなく火口基準で再計算するため、呼び出し側で設定する値を保持
    crater_wind_from_deg = (
        270.0 - math.degrees(math.atan2(v0, u0))
    ) % 360.0
    crater_transport_to_deg = (
        crater_wind_from_deg + 180.0
    ) % 360.0

    return {
        "x_m": np.asarray(xs),
        "y_m": np.asarray(ys),
        "distance_m": np.asarray(distances),
        "speed_ms": np.asarray(speeds),
        "delta_t_s": delta_t,
        "crater_u_ms": float(u0),
        "crater_v_ms": float(v0),
        "crater_speed_ms": float(speed0),
        "crater_wind_from_deg": float(crater_wind_from_deg),
        "crater_transport_to_deg": float(crater_transport_to_deg),
    }


def equation11_sigma_y(distance_m, wind_speed_ms):
    """論文式(11)：大気不安定時の風速依存型横方向拡散幅[m]。"""
    x = np.maximum(np.asarray(distance_m, dtype=float), 1.0)
    v = np.maximum(np.asarray(wind_speed_ms, dtype=float), 0.1)
    return 0.045 * (23.0 / v + 4.75) * x ** 0.86





def aggregate_model_to_tropomi_resolution(
    fine_column: np.ndarray,
    bbox: list[float],
    center_latitude: float,
    target_ns_km: float = 3.5,
    target_ew_km: float = 7.0,
):
    """高解像度モデルをTROPOMI相当の約3.5 km×7 km格子へ面積平均する。"""
    rows, cols = fine_column.shape

    north_south_km = abs(bbox[3] - bbox[1]) * 111.32
    east_west_km = (
        abs(bbox[2] - bbox[0])
        * 111.32
        * max(math.cos(math.radians(center_latitude)), 0.05)
    )

    coarse_rows = max(1, int(math.ceil(north_south_km / target_ns_km)))
    coarse_cols = max(1, int(math.ceil(east_west_km / target_ew_km)))

    row_edges = np.linspace(0, rows, coarse_rows + 1, dtype=int)
    col_edges = np.linspace(0, cols, coarse_cols + 1, dtype=int)

    coarse = np.full((coarse_rows, coarse_cols), np.nan, dtype=float)

    for i in range(coarse_rows):
        r0, r1 = row_edges[i], row_edges[i + 1]
        for j in range(coarse_cols):
            c0, c1 = col_edges[j], col_edges[j + 1]
            block = fine_column[r0:r1, c0:c1]
            if np.any(np.isfinite(block)):
                coarse[i, j] = float(np.nanmean(block))

    return coarse


def coarse_grid_latlon(
    bbox: list[float],
    coarse_shape: tuple[int, int],
):
    rows, cols = coarse_shape
    lons = np.linspace(bbox[0], bbox[2], cols)
    lats = np.linspace(bbox[3], bbox[1], rows)
    return np.meshgrid(lons, lats)


def model_rgba(model_column):
    rgba = np.zeros(model_column.shape + (4,), dtype=np.uint8)
    valid = np.isfinite(model_column) & (model_column >= SO2_BOUNDS[0])

    for index in range(len(SO2_COLORS)):
        lower = SO2_BOUNDS[index]
        upper = SO2_BOUNDS[index + 1] if index + 1 < len(SO2_BOUNDS) else np.inf
        mask = valid & (model_column >= lower) & (model_column < upper)
        color = SO2_COLORS[index]
        rgba[mask, 0] = int(color[1:3], 16)
        rgba[mask, 1] = int(color[3:5], 16)
        rgba[mask, 2] = int(color[5:7], 16)
        rgba[mask, 3] = 220

    return rgba


def calculate_quasi_steady_model(
    msm_field,
    crater_latitude: float,
    crater_longitude: float,
    bbox: list[float],
    output_size: int,
    emission_rate_t_day: float,
    maximum_distance_km: float,
    lateral_half_width_km: float,
    tropomi_peak_distance_km: float,
    peak_distance_half_width_km: float = 5.0,
    target_ns_km: float = 3.5,
    target_ew_km: float = 7.0,
):
    """式(11)を用いた準定常モデルのSO₂カラム濃度[mol/m²]を計算する。"""
    axis = calculate_main_axis(
        msm_field=msm_field,
        maximum_distance_km=maximum_distance_km,
        nominal_step_m=100.0,
    )

    # 表示格子
    lons = np.linspace(bbox[0], bbox[2], output_size)
    lats = np.linspace(bbox[3], bbox[1], output_size)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    grid_x, grid_y = local_xy_m(
        lat_grid,
        lon_grid,
        crater_latitude,
        crater_longitude,
    )

    tree = cKDTree(np.column_stack([axis["x_m"], axis["y_m"]]))
    distance_to_axis, nearest = tree.query(
        np.column_stack([grid_x.ravel(), grid_y.ravel()]),
        k=1,
    )

    along_distance = axis["distance_m"][nearest]
    local_speed = axis["speed_ms"][nearest]

    sigma_y = equation11_sigma_y(along_distance, local_speed)

    # t/day → kg/s → mol/s
    # 1 metric ton = 1,000 kg
    emission_kg_s = emission_rate_t_day * 1_000.0 / 86_400.0
    emission_mol_s = emission_kg_s / SO2_MOLAR_MASS_KG_MOL

    # 式(8)を地表から無限高度まで積分したカラム量。
    column = (
        emission_mol_s
        / (np.sqrt(2.0 * np.pi) * sigma_y * np.maximum(local_speed, 0.1))
        * np.exp(-(distance_to_axis ** 2) / (2.0 * sigma_y ** 2))
    )

    mask = (
        (along_distance >= 100.0)
        & (along_distance <= maximum_distance_km * 1000.0)
        & (distance_to_axis <= lateral_half_width_km * 1000.0)
    )
    fine_column = np.where(mask, column, np.nan).reshape(
        output_size,
        output_size,
    )

    # 高解像度モデルをTROPOMI相当の約3.5 km（南北）×7 km（東西）へ平均化。
    coarse_column = aggregate_model_to_tropomi_resolution(
        fine_column=fine_column,
        bbox=bbox,
        center_latitude=crater_latitude,
        target_ns_km=target_ns_km,
        target_ew_km=target_ew_km,
    )

    if not np.any(np.isfinite(coarse_column)):
        raise RuntimeError(
            "TROPOMI相当格子へ平均化したモデル値がすべて欠測です。"
        )

    coarse_lon_grid, coarse_lat_grid = coarse_grid_latlon(
        bbox,
        coarse_column.shape,
    )

    # TROPOMIピークの火口からの流下距離±5 kmに相当する環状帯内で、
    # モデルのピークカラム濃度を検索する。
    coarse_distances_km = haversine_km(
        crater_latitude,
        crater_longitude,
        coarse_lat_grid,
        coarse_lon_grid,
    )

    peak_distance_min_km = max(
        0.0,
        tropomi_peak_distance_km - peak_distance_half_width_km,
    )
    peak_distance_max_km = (
        tropomi_peak_distance_km + peak_distance_half_width_km
    )

    peak_band_mask = (
        np.isfinite(coarse_column)
        & (coarse_distances_km >= peak_distance_min_km)
        & (coarse_distances_km <= peak_distance_max_km)
    )

    if not np.any(peak_band_mask):
        raise RuntimeError(
            "TROPOMIピーク流下距離±5 kmの範囲内に、"
            "有効なモデル格子がありません。"
        )

    peak_band_column = np.where(
        peak_band_mask,
        coarse_column,
        -np.inf,
    )
    max_index = int(np.argmax(peak_band_column))
    max_row, max_col = np.unravel_index(
        max_index,
        coarse_column.shape,
    )

    axis_lat, axis_lon = xy_to_latlon(
        axis["x_m"],
        axis["y_m"],
        crater_latitude,
        crater_longitude,
    )

    return {
        "fine_column_mol_m2": fine_column,
        "column_mol_m2": coarse_column,
        "rgba": model_rgba(coarse_column),
        "axis_latitude": np.asarray(axis_lat),
        "axis_longitude": np.asarray(axis_lon),
        "axis_distance_km": axis["distance_m"] / 1000.0,
        "axis_speed_ms": axis["speed_ms"],
        "delta_t_s": axis["delta_t_s"],
        "crater_u_ms": axis["crater_u_ms"],
        "crater_v_ms": axis["crater_v_ms"],
        "crater_speed_ms": axis["crater_speed_ms"],
        "crater_wind_from_deg": axis["crater_wind_from_deg"],
        "crater_transport_to_deg": axis["crater_transport_to_deg"],
        "emission_kg_s": float(emission_kg_s),
        "emission_mol_s": float(emission_mol_s),
        "max_value": float(coarse_column[max_row, max_col]),
        "max_latitude": float(coarse_lat_grid[max_row, max_col]),
        "max_longitude": float(coarse_lon_grid[max_row, max_col]),
        "max_distance_km": float(coarse_distances_km[max_row, max_col]),
        "max_bearing_deg": float(
            bearing_deg(
                crater_latitude,
                crater_longitude,
                float(coarse_lat_grid[max_row, max_col]),
                float(coarse_lon_grid[max_row, max_col]),
            )
        ),
        "tropomi_peak_distance_km": float(tropomi_peak_distance_km),
        "peak_distance_min_km": float(peak_distance_min_km),
        "peak_distance_max_km": float(peak_distance_max_km),
        "coarse_rows": int(coarse_column.shape[0]),
        "coarse_cols": int(coarse_column.shape[1]),
        "tropomi_ns_km": float(target_ns_km),
        "tropomi_ew_km": float(target_ew_km),
        "sigma_y_10km_m": float(
            equation11_sigma_y(
                10_000.0,
                np.interp(
                    10_000.0,
                    axis["distance_m"],
                    axis["speed_ms"],
                ),
            )
        ),
    }


def add_peak_fitting_diagnostics(
    model_result: dict,
    maximum: dict,
    emission_rate_t_day: float,
):
    """モデル結果へピーク濃度比、推定放出率、流向差を追加する。"""
    model_peak = float(model_result["max_value"])
    tropomi_peak = float(maximum["value"])

    if not np.isfinite(model_peak) or model_peak <= 0.0:
        raise RuntimeError(
            "モデルピークカラム濃度が0以下のため、"
            "放出率をフィッティングできません。"
        )
    if not np.isfinite(tropomi_peak) or tropomi_peak <= 0.0:
        raise RuntimeError(
            "TROPOMIピークカラム濃度が0以下のため、"
            "放出率をフィッティングできません。"
        )

    tropomi_bearing = float(maximum["bearing_deg"])
    model_bearing = float(model_result["max_bearing_deg"])
    direction_difference = abs(
        (tropomi_bearing - model_bearing + 180.0) % 360.0 - 180.0
    )

    peak_ratio = tropomi_peak / model_peak
    emission_qa_pass = (
        tropomi_peak >= PLUME_THRESHOLD_MOL_M2
        and direction_difference
        <= MAX_DIRECTION_DIFFERENCE_DEG
    )

    fitted_flux = None
    corrected_flux = None
    qa_reasons = []

    if tropomi_peak < PLUME_THRESHOLD_MOL_M2:
        qa_reasons.append(
            "TROPOMIピークが0.001 mol/m²未満"
        )
    if direction_difference > MAX_DIRECTION_DIFFERENCE_DEG:
        qa_reasons.append(
            "TROPOMIとモデルの流向差が45°を超過"
        )

    if emission_qa_pass:
        fitted_flux = (
            float(emission_rate_t_day) * peak_ratio
        )
        corrected_flux = fitted_flux * 3.0

    model_result["tropomi_peak_value"] = tropomi_peak
    model_result["tropomi_peak_bearing_deg"] = tropomi_bearing
    model_result["peak_direction_difference_deg"] = direction_difference
    model_result["peak_ratio_tropomi_model"] = peak_ratio
    model_result["emission_qa_pass"] = emission_qa_pass
    model_result["emission_qa_reasons"] = qa_reasons
    model_result["fitted_emission_rate_t_day"] = fitted_flux
    model_result["corrected_emission_rate_t_day"] = corrected_flux
    return model_result



def add_manual_fitting_diagnostics(
    model_result: dict,
    selected_tropomi: dict,
    emission_rate_t_day: float,
):
    """
    手動選択したTROPOMI地点・濃度を用いてフィッティングする。

    手動計算では0.001 mol/m²未満や流向差45°超過でも参考値を算出し、
    QA判定は警告情報として保持する。
    """
    model_peak = float(model_result["max_value"])
    selected_value = float(selected_tropomi["value"])

    if not np.isfinite(model_peak) or model_peak <= 0.0:
        raise RuntimeError(
            "モデルピークカラム濃度が0以下のため、"
            "手動フィッティングを実行できません。"
        )
    if not np.isfinite(selected_value) or selected_value <= 0.0:
        raise RuntimeError(
            "手動指定するTROPOMIカラム濃度は0より大きくしてください。"
        )

    tropomi_bearing = float(selected_tropomi["bearing_deg"])
    model_bearing = float(model_result["max_bearing_deg"])
    direction_difference = abs(
        (tropomi_bearing - model_bearing + 180.0) % 360.0 - 180.0
    )

    peak_ratio = selected_value / model_peak
    fitted_flux = float(emission_rate_t_day) * peak_ratio
    corrected_flux = fitted_flux * 3.0

    qa_reasons = []
    if selected_value < PLUME_THRESHOLD_MOL_M2:
        qa_reasons.append(
            "手動指定濃度が0.001 mol/m²未満"
        )
    if direction_difference > MAX_DIRECTION_DIFFERENCE_DEG:
        qa_reasons.append(
            "手動選択地点とモデルの流向差が45°を超過"
        )

    model_result["tropomi_peak_value"] = selected_value
    model_result["tropomi_peak_bearing_deg"] = tropomi_bearing
    model_result["peak_direction_difference_deg"] = direction_difference
    model_result["peak_ratio_tropomi_model"] = peak_ratio
    model_result["emission_qa_pass"] = len(qa_reasons) == 0
    model_result["emission_qa_reasons"] = qa_reasons
    model_result["fitted_emission_rate_t_day"] = fitted_flux
    model_result["corrected_emission_rate_t_day"] = corrected_flux
    model_result["manual_fitting"] = True
    return model_result



def choose_optimal_candidate(candidates: list[dict]):
    """指定ルールに従って最適候補を選択する。"""
    if not candidates:
        raise RuntimeError("正常に計算できた候補がありません。")

    # 1) ピーク流向差が最小
    minimum_difference = min(
        candidate["model_result"]["peak_direction_difference_deg"]
        for candidate in candidates
    )
    tolerance = 1.0e-6
    tied = [
        candidate
        for candidate in candidates
        if abs(
            candidate["model_result"]["peak_direction_difference_deg"]
            - minimum_difference
        )
        <= tolerance
    ]
    reasons = [
        f"ピーク流向差が最小（{minimum_difference:.3f}°）"
    ]

    # 2) 同値なら13時JSTに近い
    if len(tied) > 1:
        minimum_time_difference = min(
            abs(candidate["jst_hour"] - 13)
            for candidate in tied
        )
        tied = [
            candidate
            for candidate in tied
            if abs(candidate["jst_hour"] - 13)
            == minimum_time_difference
        ]
        reasons.append(
            f"13時JSTとの差が最小（{minimum_time_difference}時間）"
        )

    # 3) それでも複数なら、残った候補の火口風速平均を基準に高度を選ぶ。
    if len(tied) > 1:
        reference_speed = float(
            np.mean(
                [
                    candidate["model_result"]["crater_speed_ms"]
                    for candidate in tied
                ]
            )
        )
        if reference_speed >= 5.0:
            selected_pressure = max(
                candidate["pressure_level"] for candidate in tied
            )
            tied = [
                candidate
                for candidate in tied
                if candidate["pressure_level"] == selected_pressure
            ]
            reasons.append(
                "火口風速平均が5 m/s以上のため低高度側"
                f"（{selected_pressure} hPa）"
            )
        else:
            selected_pressure = min(
                candidate["pressure_level"] for candidate in tied
            )
            tied = [
                candidate
                for candidate in tied
                if candidate["pressure_level"] == selected_pressure
            ]
            reasons.append(
                "火口風速平均が5 m/s未満のため高高度側"
                f"（{selected_pressure} hPa）"
            )

    # 完全同値が残った場合の再現性確保
    tied.sort(
        key=lambda candidate: (
            abs(candidate["jst_hour"] - 13),
            candidate["jst_hour"],
            -candidate["pressure_level"],
        )
    )
    return tied[0], " → ".join(reasons)


def add_fixed_legend(map_object: folium.Map):
    entries = "".join(
        f"""
        <div style="display:flex;align-items:center;height:22px;">
          <span style="
            display:inline-block;
            width:26px;
            height:22px;
            background:{color};
          "></span>
          <span style="margin-left:8px;white-space:nowrap;">
            {label}
          </span>
        </div>
        """
        for color, label in zip(SO2_COLORS, SO2_LABELS)
    )

    legend_html = f"""
    <div style="
        position:fixed;
        right:18px;
        bottom:35px;
        z-index:9999;
        background:rgba(255,255,255,0.95);
        border:1px solid #999;
        border-radius:5px;
        padding:10px 12px;
        box-shadow:0 1px 6px rgba(0,0,0,0.35);
        font-size:13px;
        line-height:1.1;
    ">
      <div style="
          font-size:16px;
          font-weight:700;
          margin-bottom:6px;
      ">mol/m²</div>
      {entries}
    </div>
    """

    map_object.get_root().html.add_child(
        Element(legend_html)
    )



def _web_mercator_tile_xy(
    longitude: float,
    latitude: float,
    zoom: int,
):
    latitude = min(max(latitude, -85.05112878), 85.05112878)
    scale = 2 ** int(zoom)
    x = (longitude + 180.0) / 360.0 * scale
    latitude_radians = math.radians(latitude)
    y = (
        1.0
        - math.asinh(math.tan(latitude_radians)) / math.pi
    ) / 2.0 * scale
    return x, y


def create_so2_map_export_png(
    so2_png: bytes,
    bbox: list[float],
    crater_lat: float,
    crater_lon: float,
    maximum,
    zoom: int = 8,
) -> tuple[bytes, str | None]:
    """
    OpenStreetMap背景へSO₂画像、火口、10 km円、30 km円を重ね、
    ダウンロード用PNGを作成する。
    """
    overlay = Image.open(io.BytesIO(so2_png)).convert("RGBA")
    output_width, output_height = overlay.size

    west, south, east, north = map(float, bbox)
    x_west, y_north = _web_mercator_tile_xy(west, north, zoom)
    x_east, y_south = _web_mercator_tile_xy(east, south, zoom)

    tile_x_min = int(math.floor(x_west))
    tile_x_max = int(math.floor(x_east))
    tile_y_min = int(math.floor(y_north))
    tile_y_max = int(math.floor(y_south))

    tile_size = 256
    mosaic_width = (tile_x_max - tile_x_min + 1) * tile_size
    mosaic_height = (tile_y_max - tile_y_min + 1) * tile_size
    mosaic = Image.new(
        "RGB",
        (mosaic_width, mosaic_height),
        (230, 235, 238),
    )

    tile_error = None
    headers = {
        "User-Agent": (
            "TROPOMI-SO2-Viewer/1.0 "
            "(static export image)"
        )
    }

    for tile_x in range(tile_x_min, tile_x_max + 1):
        for tile_y in range(tile_y_min, tile_y_max + 1):
            try:
                response = requests.get(
                    (
                        "https://tile.openstreetmap.org/"
                        f"{zoom}/{tile_x}/{tile_y}.png"
                    ),
                    headers=headers,
                    timeout=20,
                )
                response.raise_for_status()
                tile = Image.open(
                    io.BytesIO(response.content)
                ).convert("RGB")
                mosaic.paste(
                    tile,
                    (
                        (tile_x - tile_x_min) * tile_size,
                        (tile_y - tile_y_min) * tile_size,
                    ),
                )
            except Exception as error:
                tile_error = str(error)

    crop_left = int(round((x_west - tile_x_min) * tile_size))
    crop_top = int(round((y_north - tile_y_min) * tile_size))
    crop_right = int(round((x_east - tile_x_min) * tile_size))
    crop_bottom = int(round((y_south - tile_y_min) * tile_size))

    crop_right = max(crop_right, crop_left + 1)
    crop_bottom = max(crop_bottom, crop_top + 1)

    base = mosaic.crop(
        (crop_left, crop_top, crop_right, crop_bottom)
    ).resize(
        (output_width, output_height),
        Image.Resampling.LANCZOS,
    ).convert("RGBA")

    result = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(result)

    def point_xy(latitude: float, longitude: float):
        tile_x, tile_y = _web_mercator_tile_xy(
            longitude,
            latitude,
            zoom,
        )
        px = (
            (tile_x - x_west)
            / max(x_east - x_west, 1.0e-12)
            * output_width
        )
        py = (
            (tile_y - y_north)
            / max(y_south - y_north, 1.0e-12)
            * output_height
        )
        return float(px), float(py)

    def draw_distance_circle(radius_km: float):
        circle_bbox = radius_bbox(
            crater_lat,
            crater_lon,
            radius_km,
        )
        x_left, _ = point_xy(
            crater_lat,
            circle_bbox[0],
        )
        x_right, _ = point_xy(
            crater_lat,
            circle_bbox[2],
        )
        _, y_top = point_xy(
            circle_bbox[3],
            crater_lon,
        )
        _, y_bottom = point_xy(
            circle_bbox[1],
            crater_lon,
        )

        width = max(2, round(output_width / 400))
        # PILには破線楕円がないため、2本線で視認性を確保する。
        draw.ellipse(
            (
                min(x_left, x_right),
                min(y_top, y_bottom),
                max(x_left, x_right),
                max(y_top, y_bottom),
            ),
            outline=(255, 255, 255, 255),
            width=width + 2,
        )
        draw.ellipse(
            (
                min(x_left, x_right),
                min(y_top, y_bottom),
                max(x_left, x_right),
                max(y_top, y_bottom),
            ),
            outline=(0, 0, 0, 255),
            width=width,
        )

    draw_distance_circle(10.0)
    draw_distance_circle(30.0)

    crater_x, crater_y = point_xy(
        crater_lat,
        crater_lon,
    )
    marker_radius = max(6, round(output_width / 120))
    draw.polygon(
        [
            (crater_x, crater_y - marker_radius),
            (
                crater_x - marker_radius,
                crater_y + marker_radius,
            ),
            (
                crater_x + marker_radius,
                crater_y + marker_radius,
            ),
        ],
        fill=(0, 0, 0, 255),
        outline=(255, 255, 255, 255),
    )

    if maximum is not None:
        peak_x, peak_y = point_xy(
            maximum["latitude"],
            maximum["longitude"],
        )
        peak_radius = max(5, round(output_width / 140))
        draw.ellipse(
            (
                peak_x - peak_radius,
                peak_y - peak_radius,
                peak_x + peak_radius,
                peak_y + peak_radius,
            ),
            fill=(255, 255, 0, 255),
            outline=(0, 0, 0, 255),
            width=max(2, round(output_width / 500)),
        )

    output = io.BytesIO()
    result.convert("RGB").save(
        output,
        format="PNG",
        optimize=True,
    )
    return output.getvalue(), tile_error



def build_map(
    so2_png: bytes,
    cloud_png: bytes,
    bbox: list[float],
    crater_lat: float,
    crater_lon: float,
    zoom: int,
    so2_opacity: float,
    cloud_opacity: float,
    show_cloud: bool,
    maximum,
    model_result=None,
    model_opacity: float = 0.65,
):
    m = folium.Map(
        location=[crater_lat, crater_lon],
        zoom_start=zoom,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer(
        "OpenStreetMap",
        name="OpenStreetMap",
        show=True,
    ).add_to(m)

    folium.TileLayer(
        "CartoDB positron",
        name="淡色地図",
        show=False,
    ).add_to(m)

    so2_group = folium.FeatureGroup(
        name="TROPOMI SO₂",
        show=True,
    )
    so2_image = np.asarray(
        Image.open(io.BytesIO(so2_png)).convert("RGBA")
    )
    folium.raster_layers.ImageOverlay(
        image=so2_image,
        bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
        opacity=so2_opacity,
        name="TROPOMI SO₂",
        zindex=2,
    ).add_to(so2_group)
    so2_group.add_to(m)

    cloud_group = folium.FeatureGroup(
        name="TROPOMI 雲量",
        show=show_cloud,
    )
    cloud_image = np.asarray(
        Image.open(io.BytesIO(cloud_png)).convert("RGBA")
    )
    folium.raster_layers.ImageOverlay(
        image=cloud_image,
        bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
        opacity=cloud_opacity,
        name="TROPOMI 雲量",
        zindex=3,
    ).add_to(cloud_group)
    cloud_group.add_to(m)

    if model_result is not None:
        model_group = folium.FeatureGroup(
            name="準定常モデル SO₂",
            show=True,
        )
        folium.raster_layers.ImageOverlay(
            image=model_result["rgba"],
            bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
            opacity=model_opacity,
            name="準定常モデル SO₂",
            zindex=4,
        ).add_to(model_group)
        model_group.add_to(m)

        axis_group = folium.FeatureGroup(
            name="モデルガス主軸",
            show=True,
        )
        folium.PolyLine(
            list(zip(
                model_result["axis_latitude"],
                model_result["axis_longitude"],
            )),
            color="#111111",
            weight=3,
            tooltip="MSM-P風場から計算したガス主軸",
        ).add_to(axis_group)
        axis_group.add_to(m)

        model_max_group = folium.FeatureGroup(
            name="モデル最大カラム濃度",
            show=True,
        )
        folium.Marker(
            [model_result["max_latitude"], model_result["max_longitude"]],
            tooltip="モデル最大カラム濃度",
            popup=folium.Popup(
                (
                    "<b>準定常モデル最大値</b><br>"
                    f"SO₂：{model_result['max_value']:.6g} mol/m²<br>"
                    f"火口からの距離：{model_result['max_distance_km']:.2f} km<br>"
                    f"火口からの方位：{model_result['max_bearing_deg']:.1f}°<br>"
                    f"検索範囲：{model_result['peak_distance_min_km']:.2f}"
                    f"～{model_result['peak_distance_max_km']:.2f} km<br>"
                    f"緯度：{model_result['max_latitude']:.5f}°<br>"
                    f"経度：{model_result['max_longitude']:.5f}°"
                ),
                max_width=320,
            ),
            icon=folium.DivIcon(
                html=(
                    '<div style="font-size:30px;color:#00ff00;'
                    '-webkit-text-stroke:2px black;">◆</div>'
                )
            ),
        ).add_to(model_max_group)
        model_max_group.add_to(m)

    crater_group = folium.FeatureGroup(
        name="火口位置",
        show=True,
    )
    folium.Marker(
        [crater_lat, crater_lon],
        tooltip="火口位置",
        icon=folium.DivIcon(
            html=(
                '<div style="font-size:26px;color:black;'
                'text-shadow:0 0 2px white;">▲</div>'
            )
        ),
    ).add_to(crater_group)
    crater_group.add_to(m)

    circle10_group = folium.FeatureGroup(
        name="10 km円",
        show=True,
    )
    folium.Circle(
        [crater_lat, crater_lon],
        radius=10_000,
        color="black",
        weight=2,
        dash_array="6,5",
        fill=False,
        tooltip="火口から10 km",
    ).add_to(circle10_group)
    circle10_group.add_to(m)

    circle30_group = folium.FeatureGroup(
        name="30 km円",
        show=True,
    )
    folium.Circle(
        [crater_lat, crater_lon],
        radius=30_000,
        color="black",
        weight=2,
        dash_array="6,5",
        fill=False,
        tooltip="火口から30 km",
    ).add_to(circle30_group)
    circle30_group.add_to(m)


    if maximum is not None:
        maximum_group = folium.FeatureGroup(
            name="最大カラム濃度位置（10–30 km）",
            show=True,
        )

        popup_html = f"""
        <b>最大カラム濃度（火口から10–30 km）</b><br>
        SO₂：{maximum['value']:.6g} mol/m²<br>
        火口からの距離：{maximum['distance_km']:.2f} km<br>
        方位角：{maximum['bearing_deg']:.1f}°<br>
        緯度：{maximum['latitude']:.5f}°<br>
        経度：{maximum['longitude']:.5f}°
        """

        folium.Marker(
            [maximum["latitude"], maximum["longitude"]],
            tooltip="10–30 km内の最大SO₂",
            popup=folium.Popup(
                popup_html,
                max_width=350,
            ),
            icon=folium.DivIcon(
                html=(
                    '<div style="font-size:34px;color:#ffff00;'
                    '-webkit-text-stroke:2px black;'
                    'text-shadow:0 0 3px black;">★</div>'
                )
            ),
        ).add_to(maximum_group)

        folium.PolyLine(
            [
                [crater_lat, crater_lon],
                [maximum["latitude"], maximum["longitude"]],
            ],
            color="black",
            weight=1.5,
            dash_array="4,5",
        ).add_to(maximum_group)

        maximum_group.add_to(m)

    add_fixed_legend(m)

    Fullscreen(position="topleft").add_to(m)
    MousePosition(
        position="bottomright",
        prefix="緯度・経度:",
        separator=" / ",
        num_digits=5,
    ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    return m


st.set_page_config(
    page_title="TROPOMI SO₂ Viewer",
    page_icon="🌋",
    layout="wide",
)


require_app_password()
st.title("TROPOMI SO₂ 表示ビューア")
st.caption("TROPOMI SO₂ + 京都大学生存圏研究所 MSM-P / 準定常ガス拡散モデル")

try:
    volcanoes = load_volcanoes()
except RuntimeError as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.header("表示設定")

    selected_volcano_index = st.selectbox(
        "火山・火口を選択",
        options=range(len(volcanoes)),
        format_func=lambda index: volcano_display_name(
            volcanoes[index]
        ),
    )
    preset = volcanoes[selected_volcano_index]
    volcano_name = volcano_display_name(preset)

    selected_date = st.date_input(
        "観測日（日本時間）",
        value=date.today() - timedelta(days=1),
        max_value=date.today(),
    )

    min_qa = st.slider(
        "最小QA値（SO₂、%）",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
    )

    display_radius_km = st.selectbox(
        "表示範囲（中心からの距離）",
        [40, 50, 60, 80, 100],
        index=2,
        format_func=lambda value: f"{value} km",
    )

    image_size = st.select_slider(
        "画像サイズ",
        options=[512, 768, 1024],
        value=768,
        format_func=lambda value: f"{value} × {value}",
    )

    use_connected_plume_detection = st.checkbox(
        "連結数でプルーム判定する",
        value=False,
        help=(
            "OFF：10～30 km内の判定格子で0.001 mol/m²以上の"
            "ピークがあれば採用します。"
            "ON：閾値以上の画素が8近傍で3個以上連結する場合だけ"
            "プルームとして採用します。"
        ),
    )

    crater_lat = float(preset["latitude"])
    crater_lon = float(preset["longitude"])

    st.caption(
        f"火口情報：{preset['name']} / {preset['crater']}\n\n"
        f"緯度 {crater_lat:.4f}°、経度 {crater_lon:.4f}°"
    )

    st.subheader("レイヤー設定")

    so2_opacity = st.slider(
        "SO₂レイヤーの透明度",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
    )

    show_cloud = st.checkbox(
        "雲量レイヤーを初期表示",
        value=False,
    )

    cloud_opacity = st.slider(
        "雲量レイヤーの透明度",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
    )

    st.subheader("準定常ガス拡散モデル")

    run_model_calculation = st.checkbox(
        "モデル計算を実行する",
        value=True,
        help=(
            "OFFの場合はTROPOMI SO₂と雲量だけを取得し、"
            "MSM-Pの取得・準定常モデル・放出率推定は実行しません。"
        ),
    )

    st.markdown("##### モデル平均化分解能")

    resolution_left, resolution_right = st.columns(2)

    with resolution_left:
        model_resolution_ns_km = st.number_input(
            "南北方向（km）",
            min_value=0.5,
            max_value=30.0,
            value=3.5,
            step=0.1,
            format="%.1f",
            disabled=not run_model_calculation,
        )

    with resolution_right:
        model_resolution_ew_km = st.number_input(
            "東西方向（km）",
            min_value=0.5,
            max_value=30.0,
            value=7.0,
            step=0.1,
            format="%.1f",
            disabled=not run_model_calculation,
        )

    st.caption("初期値：南北3.5 km × 東西7.0 km")

    selected_jst_hours = st.multiselect(
        "計算時刻（JST）",
        options=list(range(0, 24, 3)),
        default=[12, 15],
        format_func=lambda value: f"{value:02d}:00 JST",
        help=(
            "選択した時刻と気圧面の全組合せを計算します。"
            "MSM-Pは3時間間隔です。"
        ),
        disabled=not run_model_calculation,
    )

    selected_pressure_levels = st.multiselect(
        "MSM-P気圧面",
        options=[1000, 950, 900, 850, 800, 700, 500],
        default=[900, 850, 800],
        format_func=lambda value: f"{value} hPa",
        disabled=not run_model_calculation,
    )

    pattern_count = (
        len(selected_jst_hours) * len(selected_pressure_levels)
    )
    st.caption(f"計算パターン数：{pattern_count} / 6")

    pattern_selection_valid = (
        not run_model_calculation
        or 1 <= pattern_count <= 6
    )
    if run_model_calculation and pattern_count == 0:
        st.error("時刻と気圧面を少なくとも1つずつ選択してください。")
    elif run_model_calculation and pattern_count > 6:
        st.error("時刻×気圧面の組合せは最大6パターンです。")

    emission_rate_t_day = st.number_input(
        "仮定SO₂放出率（t/day）",
        min_value=1.0,
        max_value=100000.0,
        value=1000.0,
        step=100.0,
        disabled=not run_model_calculation,
    )

    maximum_axis_km = st.selectbox(
        "主軸計算距離",
        [30, 40, 50],
        index=2,
        format_func=lambda value: f"{value} km",
        disabled=not run_model_calculation,
    )

    lateral_half_width_km = st.selectbox(
        "主軸からの横方向表示幅",
        [10, 15, 20, 30],
        index=2,
        format_func=lambda value: f"±{value} km",
        disabled=not run_model_calculation,
    )

    model_grid_size = st.select_slider(
        "モデル内部計算格子",
        options=[256, 384, 512],
        value=384,
        format_func=lambda value: f"{value} × {value}",
        help=("この格子で内部計算した後、指定した分解能へ平均化します。"),
        disabled=not run_model_calculation,
    )

    model_opacity = st.slider(
        "モデルレイヤーの透明度",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
        disabled=not run_model_calculation,
    )

    emission_height_mode = st.radio(
        "排出高度",
        ["MSM-P気圧面高度を使用", "手入力"],
        horizontal=False,
        disabled=not run_model_calculation,
    )
    manual_emission_height_m = st.number_input(
        "手入力排出高度（m）",
        min_value=0.0,
        max_value=20000.0,
        value=2000.0,
        step=100.0,
        disabled=(
            not run_model_calculation
            or emission_height_mode != "手入力"
        ),
    )

    run = st.button(
        "データを取得",
        type="primary",
        use_container_width=True,
        disabled=not pattern_selection_valid,
    )


client_id = get_secret("CDSE_CLIENT_ID")
client_secret = get_secret("CDSE_CLIENT_SECRET")

if not client_id or not client_secret:
    st.warning(
        "CDSE_CLIENT_IDとCDSE_CLIENT_SECRETが未設定です。"
        "`.streamlit/secrets.toml`に設定してください。"
    )
    st.code(
        'CDSE_CLIENT_ID = "実際のClient ID"\n'
        'CDSE_CLIENT_SECRET = "実際のClient Secret"',
        language="toml",
    )


if run:
    if not client_id or not client_secret:
        st.stop()

    bbox = radius_bbox(
        crater_lat,
        crater_lon,
        float(display_radius_km),
    )

    try:
        with st.spinner(
            "SO₂画像・数値データ・雲量を取得しています…"
        ):
            token = get_access_token(
                client_id,
                client_secret,
            )

            so2_png = request_process(
                token=token,
                bbox=bbox,
                target_date=selected_date,
                size=image_size,
                min_qa=min_qa,
                evalscript=fixed_so2_evalscript(),
                mime_type="image/png",
            )

            # 表示・クリック値取得用の高解像度格子。
            so2_tiff = request_process(
                token=token,
                bbox=bbox,
                target_date=selected_date,
                size=image_size,
                min_qa=min_qa,
                evalscript=float_so2_evalscript(),
                mime_type="image/tiff",
            )

            # 連結判定は表示用768×768等では行わない。
            # 約3.5 km（南北）×7 km（東西）のTROPOMI相当格子を
            # 別途取得し、元画素相当のピクセル数を数える。
            detection_width, detection_height = plume_detection_grid_size(
                bbox=bbox,
                center_latitude=crater_lat,
                target_ns_km=3.5,
                target_ew_km=7.0,
            )
            plume_detection_tiff = request_process_rectangular(
                token=token,
                bbox=bbox,
                target_date=selected_date,
                width=detection_width,
                height=detection_height,
                min_qa=min_qa,
                evalscript=float_so2_evalscript(),
                mime_type="image/tiff",
            )

            cloud_png = request_process(
                token=token,
                bbox=bbox,
                target_date=selected_date,
                size=image_size,
                min_qa=0,
                evalscript=cloud_evalscript(),
                mime_type="image/png",
            )

            cloud_tiff = request_process(
                token=token,
                bbox=bbox,
                target_date=selected_date,
                size=image_size,
                min_qa=0,
                evalscript=float_cloud_evalscript(),
                mime_type="image/tiff",
            )

            so2_array, so2_transform = read_float_tiff(so2_tiff)
            plume_detection_array, plume_detection_transform = read_float_tiff(
                plume_detection_tiff
            )
            cloud_array, cloud_transform = read_float_tiff(cloud_tiff)
            if cloud_array.shape != so2_array.shape:
                raise RuntimeError("SO₂と雲量の格子形状が一致しません。")

            minimum_connected_pixels = (
                MIN_CONNECTED_PLUME_PIXELS
                if use_connected_plume_detection
                else 1
            )

            maximum = find_annulus_maximum(
                array=plume_detection_array,
                transform=plume_detection_transform,
                crater_lat=crater_lat,
                crater_lon=crater_lon,
                min_distance_km=10.0,
                max_distance_km=30.0,
                threshold_mol_m2=PLUME_THRESHOLD_MOL_M2,
                min_connected_pixels=minimum_connected_pixels,
            )
            if maximum is not None:
                maximum["detection_grid_width"] = int(detection_width)
                maximum["detection_grid_height"] = int(detection_height)
                maximum["detection_resolution_ns_km"] = 3.5
                maximum["detection_resolution_ew_km"] = 7.0

            so2_map_png, so2_map_tile_error = (
                create_so2_map_export_png(
                    so2_png=so2_png,
                    bbox=bbox,
                    crater_lat=crater_lat,
                    crater_lon=crater_lon,
                    maximum=maximum,
                    zoom=8,
                )
            )

            tropomi_resolution = {
                "ns_km": float(model_resolution_ns_km),
                "ew_km": float(model_resolution_ew_km),
                "method": "ユーザー指定",
            }
            tropomi_resolution_error = None

            model_result = None
            model_error = None
            msm_field = None
            emission_height_m = float(manual_emission_height_m)
            model_candidates = []
            model_candidate_errors = []
            optimal_selection_reason = None

            if not run_model_calculation:
                model_error = None
            elif maximum is None:
                if use_connected_plume_detection:
                    model_error = (
                        "10～30 km内で、0.001 mol/m²以上の画素が"
                        "8近傍で3画素以上連結するプルームを"
                        "検出できないため、モデル計算を実行しませんでした。"
                    )
                else:
                    model_error = (
                        "10～30 km内に0.001 mol/m²以上のSO₂ピークを"
                        "検出できないため、モデル計算を実行しませんでした。"
                    )
            else:
                for jst_hour in sorted(selected_jst_hours):
                    # 選択日JSTからUTC日時へ変換する。
                    jst_datetime = datetime.combine(
                        selected_date,
                        time(hour=int(jst_hour)),
                    )
                    utc_datetime = jst_datetime - timedelta(hours=9)
                    msm_date_utc = utc_datetime.date()
                    msm_hour_utc = utc_datetime.hour

                    for pressure_level in sorted(
                        selected_pressure_levels,
                        reverse=True,
                    ):
                        try:
                            candidate_field = fetch_msm_p_field(
                                msm_date_utc,
                                msm_hour_utc,
                                pressure_level,
                                crater_lat,
                                crater_lon,
                            )

                            candidate_emission_height_m = float(
                                manual_emission_height_m
                            )
                            if (
                                emission_height_mode
                                == "MSM-P気圧面高度を使用"
                                and candidate_field.get(
                                    "geopotential_height_m"
                                )
                                is not None
                            ):
                                candidate_emission_height_m = float(
                                    candidate_field[
                                        "geopotential_height_m"
                                    ]
                                )

                            candidate_result = calculate_quasi_steady_model(
                                msm_field=candidate_field,
                                crater_latitude=crater_lat,
                                crater_longitude=crater_lon,
                                bbox=bbox,
                                output_size=model_grid_size,
                                emission_rate_t_day=emission_rate_t_day,
                                maximum_distance_km=maximum_axis_km,
                                lateral_half_width_km=(
                                    lateral_half_width_km
                                ),
                                tropomi_peak_distance_km=(
                                    maximum["distance_km"]
                                ),
                                peak_distance_half_width_km=5.0,
                                target_ns_km=float(model_resolution_ns_km),
                                target_ew_km=float(model_resolution_ew_km),
                            )
                            candidate_result = add_peak_fitting_diagnostics(
                                candidate_result,
                                maximum,
                                emission_rate_t_day,
                            )

                            actual_jst_hour = (
                                int(candidate_field["utc_hour"]) + 9
                            ) % 24

                            model_candidates.append(
                                {
                                    "requested_jst_hour": int(jst_hour),
                                    "jst_hour": int(actual_jst_hour),
                                    "utc_hour": int(
                                        candidate_field["utc_hour"]
                                    ),
                                    "pressure_level": int(
                                        pressure_level
                                    ),
                                    "model_result": candidate_result,
                                    "msm_field": candidate_field,
                                    "emission_height_m": (
                                        candidate_emission_height_m
                                    ),
                                }
                            )
                        except Exception as error:
                            model_candidate_errors.append(
                                {
                                    "jst_hour": int(jst_hour),
                                    "pressure_level": int(
                                        pressure_level
                                    ),
                                    "error": str(error),
                                }
                            )

                if model_candidates:
                    optimal_candidate, optimal_selection_reason = (
                        choose_optimal_candidate(model_candidates)
                    )
                    model_result = optimal_candidate["model_result"]
                    msm_field = optimal_candidate["msm_field"]
                    emission_height_m = optimal_candidate[
                        "emission_height_m"
                    ]
                    optimal_jst_hour = optimal_candidate["jst_hour"]
                    optimal_pressure_level = optimal_candidate[
                        "pressure_level"
                    ]
                else:
                    model_error = (
                        "選択した全パターンでMSM-P取得または"
                        "モデル計算に失敗しました。"
                    )

        st.session_state["viewer"] = {
            "so2_png": so2_png,
            "so2_map_png": so2_map_png,
            "so2_map_tile_error": so2_map_tile_error,
            "so2_tiff": so2_tiff,
            "cloud_png": cloud_png,
            "cloud_tiff": cloud_tiff,
            "cloud_array": cloud_array,
            "array": so2_array,
            "transform": so2_transform,
            "maximum": maximum,
            "model_result": model_result,
            "model_error": model_error,
            "run_model_calculation": run_model_calculation,
            "use_connected_plume_detection": (
                use_connected_plume_detection
            ),
            "msm_field": msm_field,
            "model_candidates": model_candidates,
            "model_candidate_errors": model_candidate_errors,
            "tropomi_resolution": tropomi_resolution,
            "optimal_selection_reason": optimal_selection_reason,
            "optimal_jst_hour": (
                optimal_jst_hour if model_result is not None else None
            ),
            "optimal_pressure_level": (
                optimal_pressure_level
                if model_result is not None
                else None
            ),
            "emission_height_m": emission_height_m,
            "emission_rate_t_day": emission_rate_t_day,
            "maximum_axis_km": maximum_axis_km,
            "lateral_half_width_km": lateral_half_width_km,
            "model_grid_size": model_grid_size,
            "model_resolution_ns_km": float(model_resolution_ns_km),
            "model_resolution_ew_km": float(model_resolution_ew_km),
            "manual_fitting_result": None,
            "bbox": bbox,
            "selected_date": selected_date,
            "crater_lat": crater_lat,
            "crater_lon": crater_lon,
            "volcano_name": volcano_name,
            "zoom": 8,
        }

    except requests.Timeout:
        st.error(
            "APIの応答がタイムアウトしました。"
            "時間をおいて再度実行してください。"
        )
    except Exception as error:
        st.error(str(error))


if "viewer" not in st.session_state:
    st.info(
        "左側で条件を設定し、"
        "「データを取得」を押してください。"
    )
else:
    data = st.session_state["viewer"]

    map_object = build_map(
        so2_png=data["so2_png"],
        cloud_png=data["cloud_png"],
        bbox=data["bbox"],
        crater_lat=data["crater_lat"],
        crater_lon=data["crater_lon"],
        zoom=data["zoom"],
        so2_opacity=so2_opacity,
        cloud_opacity=cloud_opacity,
        show_cloud=show_cloud,
        maximum=data["maximum"],
        model_result=data.get("model_result"),
        model_opacity=model_opacity,
    )

    map_result = st_folium(
        map_object,
        width=None,
        height=720,
        returned_objects=["last_clicked"],
        key="tropomi_map",
    )

    left, right = st.columns(2)

    with left:
        st.subheader("クリック位置の情報")

        clicked = map_result.get("last_clicked")

        if clicked:
            click_lat = float(clicked["lat"])
            click_lon = float(clicked["lng"])

            click_distance = float(
                haversine_km(
                    data["crater_lat"],
                    data["crater_lon"],
                    click_lat,
                    click_lon,
                )
            )

            click_value = value_at_click(
                data["array"],
                data["transform"],
                click_lat,
                click_lon,
            )

            st.write(f"緯度：{click_lat:.5f}°")
            st.write(f"経度：{click_lon:.5f}°")
            st.write(
                f"火口からの距離：{click_distance:.2f} km"
            )

            if click_value is None:
                st.write("SO₂カラム濃度：欠測")
            else:
                st.write(
                    f"SO₂カラム濃度："
                    f"**{click_value:.6g} mol/m²**"
                )
        else:
            st.write(
                "地図上をクリックすると、"
                "その地点のSO₂値と火口からの距離を表示します。"
            )

    with right:
        st.subheader(
            "最大カラム濃度"
            "（火口から10～30 kmの範囲）"
        )

        maximum = data["maximum"]

        if maximum is None:
            if data.get("use_connected_plume_detection", False):
                st.warning(
                    "10～30 km内で、0.001 mol/m²以上の画素が"
                    "8近傍で3画素以上連結するプルームを"
                    "検出できませんでした。"
                )
            else:
                st.warning(
                    "10～30 km内に0.001 mol/m²以上の"
                    "SO₂ピークを検出できませんでした。"
                )
        else:
            st.metric(
                "SO₂カラム濃度",
                f"{maximum['value']:.6g} mol/m²",
            )
            st.write(
                f"火口からの距離："
                f"{maximum['distance_km']:.2f} km"
            )
            st.write(
                f"方位角：{maximum['bearing_deg']:.1f}°"
            )
            st.write(
                f"緯度：{maximum['latitude']:.5f}°"
            )
            st.write(
                f"経度：{maximum['longitude']:.5f}°"
            )
            if data.get("use_connected_plume_detection", False):
                st.write(
                    "連結ピクセル数："
                    f"**{maximum['connected_pixel_count']}**"
                )
                st.write(
                    "連結判定格子："
                    f"**{maximum.get('detection_grid_width', '?')} × "
                    f"{maximum.get('detection_grid_height', '?')}** "
                    "（約7 km東西 × 3.5 km南北）"
                )
                st.caption(
                    "約3.5 km×7 kmのTROPOMI相当格子で、"
                    "0.001 mol/m²以上の画素が3画素以上、"
                    "8近傍で連結した成分内の最大値です。"
                    "地図上の黄色い★が最大値の位置です。"
                )
            else:
                st.caption(
                    "連結数判定はOFFです。約3.5 km×7 kmの判定格子で、"
                    "10～30 km内の0.001 mol/m²以上の画素から"
                    "最大値を採用しています。"
                    "地図上の黄色い★が最大値の位置です。"
                )

    st.divider()

    st.subheader("任意地点の追加フィッティング")
    st.caption(
        "自動計算後に地図上のTROPOMI画素をクリックし、"
        "その地点と任意のカラム濃度を使ってモデルを再計算します。"
        "自動計算結果は上書きされません。"
    )

    clicked_for_manual = map_result.get("last_clicked")
    manual_click_value = None
    manual_click_lat = None
    manual_click_lon = None
    manual_click_distance = None
    manual_click_bearing = None

    if clicked_for_manual:
        manual_click_lat = float(clicked_for_manual["lat"])
        manual_click_lon = float(clicked_for_manual["lng"])
        manual_click_value = value_at_click(
            data["array"],
            data["transform"],
            manual_click_lat,
            manual_click_lon,
        )
        manual_click_distance = float(
            haversine_km(
                data["crater_lat"],
                data["crater_lon"],
                manual_click_lat,
                manual_click_lon,
            )
        )
        manual_click_bearing = float(
            bearing_deg(
                data["crater_lat"],
                data["crater_lon"],
                manual_click_lat,
                manual_click_lon,
            )
        )

    manual_candidates_source = data.get("model_candidates", [])
    manual_ready = (
        bool(manual_candidates_source)
        and manual_click_value is not None
        and manual_click_distance is not None
        and manual_click_distance > 0.0
    )

    manual_col1, manual_col2 = st.columns(2)

    with manual_col1:
        if clicked_for_manual:
            st.write(
                f"選択地点：緯度 {manual_click_lat:.5f}° / "
                f"経度 {manual_click_lon:.5f}°"
            )
            st.write(
                f"火口からの距離：{manual_click_distance:.2f} km"
            )
            st.write(
                f"火口からの方位：{manual_click_bearing:.1f}°"
            )
        else:
            st.info(
                "先に地図上でフィッティング対象の画素をクリックしてください。"
            )

    with manual_col2:
        default_manual_value = (
            float(manual_click_value)
            if manual_click_value is not None
            and np.isfinite(manual_click_value)
            and manual_click_value > 0.0
            else 0.001
        )

        # 選択ピクセルが変わった場合だけ、入力欄を新しいクリック値へ更新する。
        # 同じピクセルのままの場合は、利用者が手入力した値を維持する。
        current_click_signature = None
        if (
            manual_click_lat is not None
            and manual_click_lon is not None
        ):
            current_click_signature = (
                round(float(manual_click_lat), 6),
                round(float(manual_click_lon), 6),
            )

        previous_click_signature = st.session_state.get(
            "manual_tropomi_click_signature"
        )

        if (
            current_click_signature is not None
            and current_click_signature
            != previous_click_signature
        ):
            st.session_state["manual_tropomi_value"] = (
                default_manual_value
            )
            st.session_state[
                "manual_tropomi_click_signature"
            ] = current_click_signature
            # 新しいピクセルを選択した時点で、前回の手動結果はクリアする。
            data["manual_fitting_result"] = None
            st.session_state["viewer"] = data

        elif "manual_tropomi_value" not in st.session_state:
            st.session_state["manual_tropomi_value"] = (
                default_manual_value
            )

        manual_tropomi_value = st.number_input(
            "手動指定TROPOMIカラム濃度（mol/m²）",
            min_value=0.000001,
            max_value=1.0,
            step=0.0001,
            format="%.6f",
            key="manual_tropomi_value",
            help=(
                "選択ピクセルが変わると、その地点のカラム濃度へ"
                "自動更新されます。同じピクセルのままでは、"
                "手入力した値を維持します。"
            ),
        )

    manual_button = st.button(
        "選択した濃度で追加フィッティングを実行",
        type="secondary",
        use_container_width=True,
        disabled=not manual_ready,
    )

    if manual_button:
        manual_target = {
            "value": float(manual_tropomi_value),
            "latitude": float(manual_click_lat),
            "longitude": float(manual_click_lon),
            "distance_km": float(manual_click_distance),
            "bearing_deg": float(manual_click_bearing),
        }

        manual_candidates = []
        manual_errors = []

        with st.spinner(
            "選択地点の流下距離に合わせてモデルを再計算しています…"
        ):
            for original_candidate in manual_candidates_source:
                try:
                    candidate_field = original_candidate["msm_field"]

                    candidate_result = calculate_quasi_steady_model(
                        msm_field=candidate_field,
                        crater_latitude=data["crater_lat"],
                        crater_longitude=data["crater_lon"],
                        bbox=data["bbox"],
                        output_size=int(data["model_grid_size"]),
                        emission_rate_t_day=float(
                            data["emission_rate_t_day"]
                        ),
                        maximum_distance_km=float(
                            data["maximum_axis_km"]
                        ),
                        lateral_half_width_km=float(
                            data["lateral_half_width_km"]
                        ),
                        tropomi_peak_distance_km=float(
                            manual_target["distance_km"]
                        ),
                        peak_distance_half_width_km=5.0,
                        target_ns_km=float(
                            data["model_resolution_ns_km"]
                        ),
                        target_ew_km=float(
                            data["model_resolution_ew_km"]
                        ),
                    )

                    candidate_result = add_manual_fitting_diagnostics(
                        candidate_result,
                        manual_target,
                        float(data["emission_rate_t_day"]),
                    )

                    manual_candidates.append(
                        {
                            "requested_jst_hour": original_candidate.get(
                                "requested_jst_hour",
                                original_candidate["jst_hour"],
                            ),
                            "jst_hour": int(
                                original_candidate["jst_hour"]
                            ),
                            "utc_hour": int(
                                original_candidate.get(
                                    "utc_hour",
                                    candidate_field["utc_hour"],
                                )
                            ),
                            "pressure_level": int(
                                original_candidate["pressure_level"]
                            ),
                            "model_result": candidate_result,
                            "msm_field": candidate_field,
                            "emission_height_m": original_candidate.get(
                                "emission_height_m"
                            ),
                        }
                    )
                except Exception as error:
                    manual_errors.append(
                        {
                            "jst_hour": original_candidate.get(
                                "jst_hour"
                            ),
                            "pressure_level": original_candidate.get(
                                "pressure_level"
                            ),
                            "error": str(error),
                        }
                    )

        if manual_candidates:
            manual_optimal, manual_reason = choose_optimal_candidate(
                manual_candidates
            )
            data["manual_fitting_result"] = {
                "target": manual_target,
                "candidates": manual_candidates,
                "candidate_errors": manual_errors,
                "optimal_candidate": manual_optimal,
                "selection_reason": manual_reason,
            }
            st.session_state["viewer"] = data
            st.success("追加フィッティングが完了しました。")
            st.rerun()
        else:
            st.error(
                "選択した地点について、すべてのモデル候補の再計算に"
                "失敗しました。"
            )
            if manual_errors:
                st.dataframe(
                    manual_errors,
                    use_container_width=True,
                    hide_index=True,
                )

    manual_fitting_result = data.get("manual_fitting_result")
    if manual_fitting_result:
        manual_target = manual_fitting_result["target"]
        manual_optimal = manual_fitting_result["optimal_candidate"]
        manual_model = manual_optimal["model_result"]

        st.markdown("#### 追加フィッティング結果")

        mt1, mt2, mt3, mt4 = st.columns(4)
        mt1.metric(
            "手動指定TROPOMI濃度",
            f"{manual_target['value']:.6g} mol/m²",
        )
        mt2.metric(
            "選択地点の流下距離",
            f"{manual_target['distance_km']:.2f} km",
        )
        mt3.metric(
            "採用モデル",
            (
                f"{manual_optimal['jst_hour']:02d}:00 JST / "
                f"{manual_optimal['pressure_level']} hPa"
            ),
        )
        mt4.metric(
            "流向差",
            f"{manual_model['peak_direction_difference_deg']:.1f}°",
        )

        mf1, mf2, mf3 = st.columns(3)
        mf1.metric(
            "モデルピーク濃度",
            f"{manual_model['max_value']:.6g} mol/m²",
        )
        mf2.metric(
            "推定放出率",
            (
                f"{manual_model['fitted_emission_rate_t_day']:.1f} "
                "t/day"
            ),
        )
        mf3.metric(
            "補正放出率（×3）",
            (
                f"{manual_model['corrected_emission_rate_t_day']:.1f} "
                "t/day"
            ),
        )

        if manual_model["emission_qa_pass"]:
            st.success(
                "手動フィッティングQA：PASS "
                "（指定濃度 ≥ 0.001 mol/m²、流向差 ≤ 45°）"
            )
        else:
            st.warning(
                "手動フィッティングは参考値として算出しました。"
                "\n\n"
                + "\n\n".join(
                    f"・{reason}"
                    for reason in manual_model[
                        "emission_qa_reasons"
                    ]
                )
            )

        with st.expander("手動フィッティング候補の比較"):
            manual_rows = []
            for candidate in sorted(
                manual_fitting_result["candidates"],
                key=lambda item: (
                    item["jst_hour"],
                    -item["pressure_level"],
                ),
            ):
                candidate_model = candidate["model_result"]
                manual_rows.append(
                    {
                        "最適": (
                            "○"
                            if (
                                candidate["jst_hour"]
                                == manual_optimal["jst_hour"]
                                and candidate["pressure_level"]
                                == manual_optimal["pressure_level"]
                            )
                            else ""
                        ),
                        "時刻（JST）": (
                            f"{candidate['jst_hour']:02d}:00"
                        ),
                        "気圧面": (
                            f"{candidate['pressure_level']} hPa"
                        ),
                        "モデルピーク": (
                            f"{candidate_model['max_value']:.6g}"
                        ),
                        "流向差": (
                            f"{candidate_model['peak_direction_difference_deg']:.3f}°"
                        ),
                        "推定放出率": (
                            f"{candidate_model['fitted_emission_rate_t_day']:.1f}"
                        ),
                        "補正放出率": (
                            f"{candidate_model['corrected_emission_rate_t_day']:.1f}"
                        ),
                    }
                )
            st.dataframe(
                manual_rows,
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                manual_fitting_result["selection_reason"]
            )

        if st.button(
            "追加フィッティング結果をクリア",
            use_container_width=True,
        ):
            data["manual_fitting_result"] = None
            st.session_state["viewer"] = data
            st.rerun()

    st.divider()
    st.subheader("準定常ガス拡散モデル")

    if not data.get("run_model_calculation", True):
        st.info(
            "モデル計算はOFFです。"
            "TROPOMI SO₂・雲量・プルーム判定のみ実行しました。"
        )

    model_result = data.get("model_result")
    model_error = data.get("model_error")
    msm_field = data.get("msm_field")

    model_candidates = data.get("model_candidates", [])
    candidate_errors = data.get("model_candidate_errors", [])
    tropomi_resolution = data.get(
        "tropomi_resolution",
        {"ns_km": 3.5, "ew_km": 7.0, "method": "ユーザー指定"},
    )

    resolution1, resolution2 = st.columns(2)
    resolution1.metric(
        "モデル平均化分解能（南北）",
        f"{tropomi_resolution['ns_km']:.1f} km",
    )
    resolution2.metric(
        "モデル平均化分解能（東西）",
        f"{tropomi_resolution['ew_km']:.1f} km",
    )
    st.caption(
        "モデルは高解像度で内部計算した後、"
        "指定した分解能へ面積平均しています。"
    )

    if model_candidates:
        st.markdown("#### 計算パターンの比較")
        comparison_rows = []
        optimal_jst_hour = data.get("optimal_jst_hour")
        optimal_pressure = data.get("optimal_pressure_level")

        for candidate in sorted(
            model_candidates,
            key=lambda item: (
                item["jst_hour"],
                -item["pressure_level"],
            ),
        ):
            result = candidate["model_result"]
            comparison_rows.append(
                {
                    "最適": (
                        "○"
                        if (
                            candidate["jst_hour"] == optimal_jst_hour
                            and candidate["pressure_level"]
                            == optimal_pressure
                        )
                        else ""
                    ),
                    "時刻（JST）": (
                        f"{candidate['jst_hour']:02d}:00"
                    ),
                    "気圧面": (
                        f"{candidate['pressure_level']} hPa"
                    ),
                    "流向差": (
                        f"{result['peak_direction_difference_deg']:.3f}°"
                    ),
                    "火口風速": (
                        f"{result['crater_speed_ms']:.2f} m/s"
                    ),
                    "モデルピーク": (
                        f"{result['max_value']:.6g} mol/m²"
                    ),
                    "推定放出率": (
                        f"{result['fitted_emission_rate_t_day']:.1f} t/day"
                        if result["fitted_emission_rate_t_day"] is not None
                        else "QA不適合"
                    ),
                }
            )

        st.dataframe(
            comparison_rows,
            use_container_width=True,
            hide_index=True,
        )

        st.success(
            "最適解："
            f"**{optimal_jst_hour:02d}:00 JST・"
            f"{optimal_pressure} hPa**"
        )
        st.caption(
            "選択理由："
            f"{data.get('optimal_selection_reason', '')}"
        )

    if candidate_errors:
        with st.expander(
            f"計算できなかったパターン（{len(candidate_errors)}件）"
        ):
            for item in candidate_errors:
                st.write(
                    f"- {item['jst_hour']:02d}:00 JST・"
                    f"{item['pressure_level']} hPa："
                    f"{item['error']}"
                )

    if model_error and data.get("run_model_calculation", True):
        st.warning(
            "MSM-P取得またはモデル計算に失敗しました。"
            "TROPOMIデータのみ表示しています。\n\n"
            f"エラー内容：{model_error}"
        )
    elif model_result is not None and msm_field is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "仮定SO₂放出率",
            f"{data['emission_rate_t_day']:.0f} t/day",
        )
        m2.metric(
            "モデル最大カラム濃度",
            f"{model_result['max_value']:.6g} mol/m²",
        )
        m3.metric(
            "最大カラム濃度の方位角",
            f"{model_result['max_bearing_deg']:.1f}°",
        )
        m4.metric(
            "最大カラム濃度の流下距離",
            f"{model_result['max_distance_km']:.2f} km",
        )
        st.write(
            "最適パターン："
            f"**{data['optimal_jst_hour']:02d}:00 JST** "
            f"（{msm_field['analysis_time']}）／ "
            f"**{msm_field['pressure_level']} hPa**"
        )
        st.write(
            "モデル平均化解像度："
            f"**南北 {model_result['tropomi_ns_km']:.2f} km × "
            f"東西 {model_result['tropomi_ew_km']:.2f} km**"
        )
        st.markdown("#### 火口位置のMSM-P風")

        wind1, wind2, wind3, wind4 = st.columns(4)

        wind1.metric(
            "風速",
            f"{model_result['crater_speed_ms']:.2f} m/s",
        )
        wind2.metric(
            "風向（吹いてくる方向）",
            f"{model_result['crater_wind_from_deg']:.1f}°",
        )
        wind3.metric(
            "移流方向",
            f"{model_result['crater_transport_to_deg']:.1f}°",
        )
        wind4.metric(
            "u / v成分",
            (
                f"{model_result['crater_u_ms']:.2f} / "
                f"{model_result['crater_v_ms']:.2f} m/s"
            ),
        )
        st.write(
            f"放出率換算：**{model_result['emission_kg_s']:.4f} kg/s** "
            f"= **{model_result['emission_mol_s']:.2f} mol/s**"
        )
        st.write(
            "モデルピーク検索範囲："
            f"**{model_result['peak_distance_min_km']:.2f}～"
            f"{model_result['peak_distance_max_km']:.2f} km** "
            "（TROPOMIピーク流下距離 "
            f"{model_result['tropomi_peak_distance_km']:.2f} km ±5 km）"
        )
        st.divider()
        st.markdown("#### ピークカラム濃度フィッティングによる放出率")

        fit1, fit2, fit3, fit4 = st.columns(4)

        fit1.metric(
            "TROPOMIピーク",
            f"{model_result['tropomi_peak_value']:.6g} mol/m²",
        )
        fit2.metric(
            "モデルピーク",
            f"{model_result['max_value']:.6g} mol/m²",
        )
        if model_result["emission_qa_pass"]:
            fit3.metric(
                "推定放出率",
                (
                    f"{model_result['fitted_emission_rate_t_day']:.1f} "
                    "t/day"
                ),
            )
            fit4.metric(
                "補正放出率（×3）",
                (
                    f"{model_result['corrected_emission_rate_t_day']:.1f} "
                    "t/day"
                ),
            )
        else:
            fit3.metric("推定放出率", "算出対象外")
            fit4.metric("補正放出率（×3）", "算出対象外")

        st.write(
            "ピーク濃度比（TROPOMI／モデル）："
            f"**{model_result['peak_ratio_tropomi_model']:.3f}**"
        )
        direction1, direction2, direction3 = st.columns(3)

        direction1.metric(
            "TROPOMIピーク流向",
            f"{model_result['tropomi_peak_bearing_deg']:.1f}°",
        )
        direction2.metric(
            "モデルピーク流向",
            f"{model_result['max_bearing_deg']:.1f}°",
        )
        direction3.metric(
            "流向差の絶対値",
            (
                f"{model_result['peak_direction_difference_deg']:.1f}°"
            ),
            help=(
                "|TROPOMIピーク流向－モデルピーク流向|を、"
                "0°／360°を考慮した最小角度差で表示します。"
            ),
        )
        if model_result["emission_qa_pass"]:
            st.success(
                "放出率推定QA：PASS "
                "（ピーク ≥ 0.001 mol/m²、流向差 ≤ 45°）"
            )
        else:
            st.warning(
                "放出率推定QA：FAIL\n\n"
                + "\n\n".join(
                    f"・{reason}"
                    for reason in model_result["emission_qa_reasons"]
                )
            )
        st.caption(
            "ピークが0.001 mol/m²以上、連結3画素以上、"
            "かつ流向差が45°以下の場合のみ放出率を算出します。"
            "推定放出率 ＝ 入力した仮定放出率 × "
            "（TROPOMIピークカラム濃度／モデルピークカラム濃度）。"
            "モデル濃度が放出率に比例することを利用しています。"
            "補正放出率は、この推定放出率を3倍した値です。"
        )
        st.caption(
            "横方向拡散幅は論文式(11) "
            "σy = 0.045 × (23/V′ + 4.75) × x^0.86 を使用しています。"
            "表示するSO₂カラム濃度は式(8)を鉛直方向に解析積分し、""ユーザーが指定した分解能へ面積平均した値です。"
            "このため、地表から無限高度までの全カラムでは排出高度は計算値に影響しません。"
        )

    if data.get("so2_map_tile_error"):
        st.caption(
            "保存画像の背景地図タイルを一部取得できなかった可能性があります。"
            "SO₂、火口、10 km円、30 km円は保存画像へ描画されています。"
        )

    download_left, download_right = st.columns(2)

    with download_left:
        st.download_button(
            "SO₂表示画像を保存（地図・10/30 km円付きPNG）",
            data=data["so2_map_png"],
            file_name=(
                f"tropomi_so2_"
                f"{data['selected_date'].isoformat()}.png"
            ),
            mime="image/png",
            use_container_width=True,
        )

    with download_right:
        st.download_button(
            "SO₂数値データを保存（GeoTIFF）",
            data=data["so2_tiff"],
            file_name=(
                f"tropomi_so2_"
                f"{data['selected_date'].isoformat()}.tif"
            ),
            mime="image/tiff",
            use_container_width=True,
        )

with st.expander("表示・最大値・風データについて"):
    st.markdown(
        """
- SO₂レイヤーは指定された固定階級で色分けしています。
- 0.0004 mol/m²未満は透明表示です。
- 最大値は、QA条件を適用したSO₂数値GeoTIFFから計算します。
- 各格子中心と火口の大円距離を求め、10 km以上30 km以下の格子だけを対象にしています。
- 表示値はSentinel Hubで地図格子へ再サンプリングされた値です。
- TROPOMI Level-2 NetCDFの元ピクセル値と完全に同一ではありません。
- 準定常モデルの横方向拡散幅は論文式(11)の大気不安定条件を使用します。
- MSM-Pは3時間間隔のため、選択時刻に最も近い解析時刻の風場を固定して計算します。
- モデルカラム濃度は式(8)の鉛直解析積分値です。
- モデルは高解像度で内部計算した後、約3.5 km×7 kmのTROPOMI相当格子へ面積平均して表示します。
- 放出率は1 t = 1,000 kgとして、t/day → kg/s → mol/sの順に変換します。
- モデルピークは、10～30 km内で検出したTROPOMIピークの火口からの流下距離±5 kmの環状帯内で検索します。
- モデル計算結果の上段には、仮定SO₂放出率、モデル最大カラム濃度、最大カラム濃度の方位角、最大カラム濃度の流下距離を表示します。
- 最大6つの時刻×気圧面パターンを計算し、ピーク流向差が最小の候補を最適解とします。
- 流向差が同じ場合は13時JSTに近い時刻を優先します。
- さらに同値の場合は、残った候補の火口風速平均が5 m/s以上なら高い気圧面（低高度側）、5 m/s未満なら低い気圧面（高高度側）を選択します。
- ピークフィッティング放出率は、入力放出率にTROPOMIピーク／モデルピークの比を乗じて算出します。
- 自動計算後は、地図で選択した任意地点と任意のカラム濃度を使い、同じ風候補で追加フィッティングできます。
- 手動追加フィッティングは自動結果を上書きせず、別枠の参考値として表示します。
- 補正放出率はピークフィッティング放出率の3倍として表示します。
- TROPOMIピーク流向とモデルピーク流向は、火口から各ピーク地点への方位角（北0°、東90°、南180°、西270°）です。
- 流向差の絶対値は、0°／360°を考慮した0～180°の最小角度差です。
- 火口位置の風は、火口から半径10 km以内のMSM-P格子を距離二乗逆数で内挿して算出します。
- 風向は風が吹いてくる方向、移流方向はその反対方向です。
        """
    )


st.divider()
st.caption(
    "TROPOMI SO₂ data: Copernicus Sentinel-5P / "
    "Wind data: JMA MSM-P archive provided by RISH, Kyoto University. "
    "This application provides simplified analytical results and "
    "does not replace official volcanic monitoring information."
)
