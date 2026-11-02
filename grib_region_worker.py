"""MSM-P FH0 GRIB2 regional extraction worker. Run only as a subprocess."""
import argparse
import math
from pathlib import Path
import numpy as np

def _msm_diag(message):
    pass

def _grib_memory_checkpoint(*args, **kwargs):
    pass

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

def _read_grib_fields_levels(path: Path, pressure_levels: tuple):
    """GRIB2を一度だけ走査して、指定した全気圧面のFH0をまとめて抽出する。"""
    import eccodes as ec
    levels = tuple(sorted(set(int(level) for level in pressure_levels)))
    required = {(level, name) for level in levels for name in ("u", "v", "gh")}
    selected = {}
    grids = {}
    scanned = 0
    _msm_diag(f"{path.name} {levels}hPa 一括GRIB2走査開始")
    _grib_memory_checkpoint("GRIB走査直前")
    _grib_memory_checkpoint("一括GRIB走査直前")
    ec.codes_grib_multi_support_on()
    with path.open("rb") as fh:
        while True:
            gid = ec.codes_grib_new_from_file(fh)
            if gid is None:
                break
            scanned += 1
            try:
                try:
                    name = str(ec.codes_get(gid, "shortName"))
                    if name not in ("u", "v", "gh"):
                        continue
                    kind = str(ec.codes_get(gid, "typeOfLevel"))
                    if kind not in ("isobaricInhPa", "isobaricInPa"):
                        continue
                    level = float(ec.codes_get(gid, "level"))
                    if kind == "isobaricInPa":
                        level /= 100
                    rounded = round(level)
                    if abs(level - rounded) > 0.01 or rounded not in levels:
                        continue
                    step = float(ec.codes_get(gid, "endStep", ktype=float))
                    unit = str(ec.codes_get(gid, "stepUnits"))
                    if unit in ("m", "min"):
                        step /= 60
                    elif unit in ("s", "sec"):
                        step /= 3600
                    elif unit in ("d", "day"):
                        step *= 24
                    if abs(step) > 1e-6 or (rounded, name) in selected:
                        continue
                    ni = int(ec.codes_get(gid, "Ni"))
                    nj = int(ec.codes_get(gid, "Nj"))
                    values = np.asarray(ec.codes_get_array(gid, "values"), dtype=np.float32)
                    if values.size != ni * nj:
                        raise RuntimeError("GRIB2の格子点数が一致しません。")
                    if rounded not in grids:
                        lat = np.asarray(ec.codes_get_array(gid, "latitudes"), dtype=np.float32).reshape(nj, ni)
                        lon = np.asarray(ec.codes_get_array(gid, "longitudes"), dtype=np.float32).reshape(nj, ni)
                        grids[rounded] = (lat, lon)
                    selected[(rounded, name)] = values.reshape(nj, ni)
                    if required.issubset(selected):
                        break
                except (KeyError, ValueError, TypeError):
                    continue
            finally:
                ec.codes_release(gid)
            if scanned % 150 == 0:
                _msm_diag(f"一括GRIB2走査 {scanned}メッセージ")
    _grib_memory_checkpoint("一括GRIB走査直後・全GRIBハンドルrelease済", selected=selected)
    _msm_diag(f"一括GRIB2走査完了 {scanned}メッセージ / {len(selected)}場")
    for level in levels:
        for name in ("u", "v"):
            if (level, name) not in selected:
                raise RuntimeError(f"{level}hPaのFH0 {name}が見つかりません。")
    return selected, grids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--levels', required=True)
    parser.add_argument('--lat', type=float, required=True)
    parser.add_argument('--lon', type=float, required=True)
    args = parser.parse_args()
    levels = tuple(int(v) for v in args.levels.split(','))
    fields, grids = _read_grib_fields_levels(Path(args.input), levels)
    arrays = {}
    for level in levels:
        lats, lons = grids[level]
        region = ((lats >= args.lat - 1.2) & (lats <= args.lat + 1.2) &
                  (lons >= args.lon - 1.5) & (lons <= args.lon + 1.5))
        if not np.any(region):
            raise RuntimeError(f'{level}hPa: 火口周辺にMSMの格子点がありません。')
        rows, cols = np.where(region)
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        lat_grid = lats[r0:r1, c0:c1].copy()
        lon_grid = lons[r0:r1, c0:c1].copy()
        arrays[f'{level}_lat'] = lat_grid
        arrays[f'{level}_lon'] = lon_grid
        for name in ('u', 'v', 'gh'):
            if (level, name) in fields:
                arrays[f'{level}_{name}'] = fields[(level, name)][r0:r1, c0:c1].copy()
    # Save only small regional arrays, never the full decoded GRIB grids.
    with open(args.output, 'wb') as fh:
        np.savez(fh, **arrays)

if __name__ == '__main__':
    main()
