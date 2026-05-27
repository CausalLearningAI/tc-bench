"""Stage 04 — crop fixed-size IR windows around each best-track fix (§3).

For every storm in the cleaned per-agency tables (stage 03), open the matching
GridSat-B1 frame (stage 02) and cut a 224x224 window centred on the storm. The
per-frame windows are stacked into one trajectory netCDF per cyclone, in the
exact schema the consolidation stage (05) consumes:

    irwin_cdr        (time, lat, lon)  brightness temperature, attr valid_range
    Min pressure mb  (time,)           agency-native minimum central pressure
    Max wind kts     (time,)           agency-native maximum sustained wind
    LAT center       (time,)           storm-centre latitude
    LON center       (time,)           storm-centre longitude
    coord time       (time,)           "%Y.%m.%d.%H" strings on the 3-hour grid

Output: ``$CROPPED_DIR/{agency}/{year}_{name}.nc`` plus a ``.done`` marker
written by the SLURM wrapper.
"""

import argparse
import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

AGENCY_CAPITAL = {
    "atcf": "USA", "bom": "BOM", "hurdat_atl": "USA", "hurdat_epa": "USA",
    "nadi": "NADI", "newdelhi": "NEWDELHI", "reunion": "REUNION",
    "tokyo": "TOKYO", "wellington": "WELLINGTON",
}
HALF_WIDTH_POINTS = 112          # 224x224 windows (~1700 km at GridSat-B1 res)
BT_MIN, BT_MAX = 140.0, 375.0    # valid brightness-temperature range (Kelvin)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def iso_to_stamp(iso_time: str) -> str:
    """``2005-08-23 00:00:00`` -> ``2005.08.23.00`` (GridSat filename stamp)."""
    return iso_time.split(":")[0].replace("-", ".").replace(" ", ".")


def crop_window(frame: xr.DataArray, lat_c: float, lon_c: float) -> np.ndarray:
    """Cut a 224x224 window centred on ``(lat_c, lon_c)`` (lon wraps at 360)."""
    lat, lon = frame.lat.values, frame.lon.values
    i_lat = int(np.argmin(np.abs(lat - lat_c)))
    i_lon = int(np.argmin(np.abs(lon - lon_c)))
    lat_idx = np.arange(i_lat - HALF_WIDTH_POINTS, i_lat + HALF_WIDTH_POINTS)
    lon_idx = np.arange(i_lon - HALF_WIDTH_POINTS, i_lon + HALF_WIDTH_POINTS) % len(lon)
    window = frame.isel(
        lat=xr.DataArray(lat_idx, dims="lat"),
        lon=xr.DataArray(lon_idx, dims="lon"),
    )
    return window.values


def fill_nans(frame: np.ndarray, max_nan_frac: float = 0.02) -> np.ndarray | None:
    """Nearest-neighbour fill small NaN gaps; reject frames with large holes.

    Returns the filled frame, or ``None`` when the farthest NaN-to-valid
    distance exceeds ``max_nan_frac`` of the frame width (frame unusable).
    """
    if not np.isnan(frame).any():
        return np.clip(frame, BT_MIN, BT_MAX)

    h, w = frame.shape
    yy, xx = np.mgrid[0:h, 0:w]
    valid = ~np.isnan(frame)
    if not valid.any():
        return None

    valid_pts = np.column_stack([yy[valid], xx[valid]])
    nan_pts = np.column_stack([yy[~valid], xx[~valid]])
    dist, _ = cKDTree(valid_pts).query(nan_pts, k=1)
    if dist.max() > max_nan_frac * w:
        return None

    filled = frame.copy()
    filled[~valid] = griddata(valid_pts, frame[valid], nan_pts, method="nearest")
    return np.clip(filled, BT_MIN, BT_MAX)


def _crop_cyclone(task: tuple) -> str:
    """Crop one cyclone's trajectory and write it; returns a status string."""
    df_cyc, year, name, var_press, var_wind, gridsat_dir, out_path = task
    out_path = Path(out_path)
    if out_path.exists():
        return f"[skip] {out_path.name} exists"
    gridsat_dir = Path(gridsat_dir)

    frames, pres, wind, lat_c, lon_c, stamps = [], [], [], [], [], []
    for _, row in df_cyc.sort_values("ISO_TIME").iterrows():
        stamp = iso_to_stamp(row["ISO_TIME"])
        nc = gridsat_dir / year / f"GRIDSAT-B1.{stamp}.v02r01.nc"
        if not nc.exists():
            continue
        try:
            ds = xr.open_dataset(nc)[["irwin_cdr"]]
            window = crop_window(ds["irwin_cdr"].isel(time=0), row["LAT"], row["LON"])
            ds.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed %s %s: %s", name, stamp, exc)
            continue
        if window.shape != (2 * HALF_WIDTH_POINTS, 2 * HALF_WIDTH_POINTS):
            continue
        filled = fill_nans(window)
        if filled is None:
            continue
        frames.append(filled.astype(np.float32))
        pres.append(row[var_press])
        wind.append(row[var_wind])
        lat_c.append(row["LAT"])
        lon_c.append(row["LON"])
        stamps.append(stamp)

    if len(frames) < 2:
        return f"[drop] {year}_{name}: <2 usable frames"

    ds_out = xr.Dataset(
        data_vars={
            "irwin_cdr": (("time", "lat", "lon"), np.stack(frames)),
            "Min pressure mb": (("time",), np.array(pres, dtype=np.float32)),
            "Max wind kts": (("time",), np.array(wind, dtype=np.float32)),
            "LAT center": (("time",), np.array(lat_c, dtype=np.float32)),
            "LON center": (("time",), np.array(lon_c, dtype=np.float32)),
        },
        coords={
            "time": stamps,
            "lat": np.arange(2 * HALF_WIDTH_POINTS),
            "lon": np.arange(2 * HALF_WIDTH_POINTS),
        },
    )
    ds_out["irwin_cdr"].attrs["valid_range"] = [BT_MIN, BT_MAX]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(
        out_path,
        encoding={"irwin_cdr": {"zlib": True, "complevel": 5, "dtype": "float32"}},
    )
    return f"[done] {out_path.name} ({len(frames)} frames)"


def build_tasks(
    input_dir: Path, gridsat_dir: Path, output_dir: Path,
    years: list[str], agencies: list[str], only_cyclone: str,
) -> list[tuple]:
    """Enumerate one cropping task per (agency, year, cyclone) in the subset."""
    tasks: list[tuple] = []
    for agency in agencies:
        csv = input_dir / f"dataset_ibtracs_basic_cols_{agency}.csv"
        if not csv.exists():
            logger.warning("[skip] no preprocessed table for %s", agency)
            continue
        capital = AGENCY_CAPITAL[agency]
        var_press, var_wind = f"{capital}_PRES mb", f"{capital}_WIND kts"
        df = pd.read_csv(csv)
        df["LON"] = (df["LON"] + 180) % 360 - 180  # normalise to [-180, 180)
        df["year"] = df["ISO_TIME"].str.slice(0, 4)
        for year, df_year in df.groupby("year"):
            if years and year not in years:
                continue
            for name, df_cyc in df_year.groupby("NAME"):
                if only_cyclone and only_cyclone.lower() not in str(name).lower():
                    continue
                out_path = output_dir / agency / f"{year}_{name}.nc"
                tasks.append((df_cyc, year, name, var_press, var_wind,
                              str(gridsat_dir), str(out_path)))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="$PREPROCESSED_DIR (per-agency CSVs from stage 03).")
    parser.add_argument("--gridsat_dir", type=Path, required=True,
                        help="$GRIDSAT_DIR (downloaded frames from stage 02).")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="$CROPPED_DIR (per-cyclone trajectory netCDFs).")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--years", default="", help="Space-separated; empty => all.")
    parser.add_argument("--agencies", default="", help="Space-separated; empty => all.")
    parser.add_argument("--only-cyclone", dest="only_cyclone", default="",
                        help="Substring match on cyclone NAME; empty => all.")
    args = parser.parse_args()

    agencies = args.agencies.split() or list(AGENCY_CAPITAL)
    tasks = build_tasks(args.input_dir, args.gridsat_dir, args.output_dir,
                        args.years.split(), agencies, args.only_cyclone)
    if not tasks:
        raise SystemExit("[error] no cyclones matched the requested subset")

    logger.info("Cropping %d cyclones with %d worker(s)", len(tasks), args.num_workers)
    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            results = pool.map(_crop_cyclone, tasks)
    else:
        results = [_crop_cyclone(t) for t in tasks]
    for line in results:
        logger.info(line)


if __name__ == "__main__":
    main()
