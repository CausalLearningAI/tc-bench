"""Stage 02 — download GridSat-B1 brightness-temperature netCDFs (§3).

Only the three-hourly global frames that coincide with a best-track fix are
fetched (the dataset never uses cloud-free background frames). The set of
timestamps is derived from IbTRACS, optionally narrowed to a subset of years
and/or a single cyclone for partial / smoke-test builds.

Output: ``$GRIDSAT_DIR/{year}/GRIDSAT-B1.{YYYY.MM.DD.HH}.v02r01.nc`` plus a
``$GRIDSAT_DIR/.done`` marker written by the SLURM wrapper.
"""

import argparse
from pathlib import Path

import pandas as pd
import requests
import wget
from bs4 import BeautifulSoup

GRIDSAT_URL = (
    "https://www.ncei.noaa.gov/data/"
    "geostationary-ir-channel-brightness-temperature-gridsat-b1/access"
)


def iso_to_gridsat_stamp(iso_time: pd.Series) -> pd.Series:
    """Map IbTRACS ``ISO_TIME`` (``YYYY-MM-DD HH:MM:SS``) to ``YYYY.MM.DD.HH``."""
    stamp = iso_time.str.split(":").str[0]          # drop minutes/seconds
    stamp = stamp.str.replace("-", ".", regex=False)
    stamp = stamp.str.replace(" ", ".", regex=False)
    return stamp


def select_timestamps(
    ibtracs_csv: Path, years: list[str], only_cyclone: str
) -> dict[str, set[str]]:
    """Return ``{year: {timestamp, ...}}`` for the requested subset.

    Empty ``years`` means every year present in the CSV; empty
    ``only_cyclone`` means every cyclone.
    """
    df = pd.read_csv(ibtracs_csv, usecols=["ISO_TIME", "NAME"], dtype=str)
    df = df.iloc[1:]  # drop the units row that IbTRACS prepends
    df = df.dropna(subset=["ISO_TIME"])

    if only_cyclone:
        df = df[df["NAME"].fillna("").str.contains(only_cyclone, case=False)]

    df = df.assign(stamp=iso_to_gridsat_stamp(df["ISO_TIME"]))
    df = df.assign(year=df["stamp"].str.slice(0, 4))

    if years:
        df = df[df["year"].isin(years)]

    return {y: set(g["stamp"]) for y, g in df.groupby("year")}


def download_year(year: str, stamps: set[str], out_dir: Path) -> None:
    """Download every GridSat-B1 file for ``year`` whose stamp is in ``stamps``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index = f"{GRIDSAT_URL}/{year}/"
    resp = requests.get(index, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.lower().endswith(".nc"):
            continue
        if not any(stamp in href for stamp in stamps):
            continue
        fname = href.split("/")[-1]
        target = out_dir / fname
        if target.exists():
            continue
        link = href if href.startswith("http") else index + href
        try:
            print(f"[download] {fname}")
            wget.download(link, out=str(out_dir))
            print()
        except Exception as exc:  # noqa: BLE001 — log and continue
            print(f"[warn] failed {fname}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--ibtracs_csv", type=Path, required=True)
    parser.add_argument(
        "--years", default="", help="Space-separated years; empty => all."
    )
    parser.add_argument(
        "--only-cyclone",
        dest="only_cyclone",
        default="",
        help="Substring match on cyclone NAME; empty => all.",
    )
    args = parser.parse_args()

    years = args.years.split()
    per_year = select_timestamps(args.ibtracs_csv, years, args.only_cyclone)
    if not per_year:
        raise SystemExit("[error] no timestamps matched the requested subset")

    for year, stamps in sorted(per_year.items()):
        print(f"[year] {year}: {len(stamps)} timestamps")
        download_year(year, stamps, args.output_dir / year)

    print(f"[done] GridSat netCDFs in {args.output_dir}")


if __name__ == "__main__":
    main()
