"""Stage 03 — preprocess IbTRACS best tracks (§3).

Cleans the raw best-track CSV and writes one tidy per-agency table that the
cropping stage consumes:

* restrict to the strict three-hourly grid (00, 03, … 21 UTC),
* assign each storm to its reporting agency,
* trim leading/trailing NaNs in wind/pressure and linearly interpolate gaps,
* disambiguate the ``UNNAMED`` storms so each ``(agency, name)`` is unique.

Output: ``$PREPROCESSED_DIR/dataset_ibtracs_basic_cols_{agency}.csv`` plus a
``.done`` marker written by the SLURM wrapper.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# IbTRACS reporting agencies and the column prefix each one uses for its
# native wind/pressure estimates (e.g. hurdat_atl -> "USA_PRES mb").
AGENCIES = ["atcf", "bom", "hurdat_atl", "hurdat_epa", "nadi",
            "newdelhi", "reunion", "tokyo", "wellington"]
CAPITALS = ["USA", "BOM", "USA", "USA", "NADI",
            "NEWDELHI", "REUNION", "TOKYO", "WELLINGTON"]
AGENCY_CAPITAL = dict(zip(AGENCIES, CAPITALS))

VALID_TIMES = [f"{h:02d}:00:00" for h in range(0, 24, 3)]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_ibtracs(path: Path) -> pd.DataFrame:
    """Load IbTRACS, folding the units row into the column names."""
    df = pd.read_csv(path)
    first_row = df.iloc[0].astype(str).str.strip()
    df.columns = [
        f"{col} {unit}".strip() if "degrees" not in unit else col
        for col, unit in zip(df.columns, first_row)
    ]
    return df.iloc[1:].reset_index(drop=True)


def enforce_3h_resolution(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only fixes on the strict 3-hourly grid and record per-storm gaps."""
    df = df.loc[df["ISO_TIME"].str[-8:].isin(VALID_TIMES)].copy()
    iso_dt = pd.to_datetime(df["ISO_TIME"])
    df["time_gap"] = (
        iso_dt.groupby(df["SID"]).diff().dt.total_seconds() / 3600
    )
    return df


def clean_and_interpolate_track(
    track: pd.DataFrame,
    cols: tuple[str, ...] = ("WMO_WIND kts", "WMO_PRES mb"),
) -> pd.DataFrame | None:
    """Trim leading/trailing NaNs in wind/pressure, then linearly interpolate.

    Returns ``None`` if the storm has no valid wind/pressure data at all.
    """
    track = track.sort_values("ISO_TIME")
    valid = track[list(cols)].notna().any(axis=1)
    if not valid.any():
        return None
    first, last = valid.idxmax(), valid[::-1].idxmax()
    track = track.loc[first:last].copy()
    track[list(cols)] = track[list(cols)].interpolate(method="linear")
    return track


def split_by_agency(df: pd.DataFrame, agency: str, capital: str) -> pd.DataFrame:
    """Select storms reported by ``agency`` and clean their tracks."""
    lat_lon = list(df.columns[df.columns.str.contains("LAT|LON")])
    wind_press = list(df.columns[df.columns.str.contains("WIND|PRES")])
    for col in wind_press + lat_lon:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["WMO_AGENCY"] = df["WMO_AGENCY"].astype(str).str.strip().replace("", np.nan)
    agency_per_sid = df.groupby("SID")["WMO_AGENCY"].apply(
        lambda x: x.dropna().unique()
    )
    valid_sid = agency_per_sid[agency_per_sid.map(len) == 1]
    sid_to_agency = valid_sid.map(lambda x: x[0])
    df = df[df["SID"].isin(sid_to_agency.index)].copy()
    df["WMO_AGENCY"] = df["SID"].map(sid_to_agency)

    sub = df[df["WMO_AGENCY"] == agency]
    cols = (
        ["SID", "SEASON Year", "NUMBER", "BASIN", "SUBBASIN",
         "NAME", "ISO_TIME", "NATURE", "LAT", "LON", "time_gap"]
        + list(sub.columns[sub.columns.str.contains(capital)])
        + list(sub.columns[sub.columns.str.contains("WMO")])
    )
    sub = sub[cols]
    sub = (
        sub.groupby("SID", group_keys=False)
        .apply(clean_and_interpolate_track)
        .dropna(how="all")
    )
    return sub.reset_index(drop=True)


def rename_unnamed_cyclones(df: pd.DataFrame, agency: str) -> pd.DataFrame:
    """Give every ``UNNAMED`` storm a unique name so files don't collide."""
    unnamed = df[df["NAME"] == "UNNAMED"].copy()
    named = df[df["NAME"] != "UNNAMED"].copy()
    for i, sid in enumerate(unnamed["SID"].unique()):
        unnamed.loc[unnamed["SID"] == sid, "NAME"] = f"UNNAMED_{i}_{agency}"
    return pd.concat([named, unnamed], ignore_index=True)


def preprocess_agency(
    df: pd.DataFrame, agency: str, years: list[str], out_dir: Path
) -> None:
    """Build and save the cleaned per-agency table (idempotent)."""
    out_csv = out_dir / f"dataset_ibtracs_basic_cols_{agency}.csv"
    if out_csv.exists():
        logger.info("[skip] %s already exists", out_csv.name)
        return
    sub = split_by_agency(df.copy(), agency, AGENCY_CAPITAL[agency])
    sub = rename_unnamed_cyclones(sub, agency)
    if years:
        sub = sub[sub["ISO_TIME"].str.slice(0, 4).isin(years)]
    if sub.empty:
        logger.warning("[empty] %s has no storms in the requested years", agency)
        return
    sub.to_csv(out_csv, index=False)
    logger.info("[done] %s (%d fixes, %d storms)",
                out_csv.name, len(sub), sub["NAME"].nunique())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ibtracs_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    # Accepted for wrapper compatibility; preprocessing is light and serial.
    parser.add_argument("--gridsat_dir", type=Path, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--years", default="", help="Space-separated; empty => all.")
    parser.add_argument("--agencies", default="", help="Space-separated; empty => all.")
    args = parser.parse_args()

    years = args.years.split()
    agencies = args.agencies.split() or AGENCIES
    unknown = set(agencies) - set(AGENCIES)
    if unknown:
        raise SystemExit(f"[error] unknown agencies: {sorted(unknown)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading IbTRACS from %s", args.ibtracs_csv)
    df = enforce_3h_resolution(load_ibtracs(args.ibtracs_csv))

    for agency in agencies:
        preprocess_agency(df, agency, years, args.output_dir)


if __name__ == "__main__":
    main()
