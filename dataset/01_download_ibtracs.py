"""Stage 01 — download the IbTRACS v4r01 best-track CSV (§3).

Downloads the multi-agency best-track archive (since 1980) to ``--output``.
Idempotent: skips the download if the target already exists.
"""

import argparse
from pathlib import Path

import wget

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.since1980.list.v04r01.csv"
)


def download_ibtracs(output: Path) -> None:
    """Download the IbTRACS CSV to ``output`` (no-op if it already exists)."""
    if output.exists():
        print(f"[skip] IbTRACS already present: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {IBTRACS_URL} -> {output}")
    wget.download(IBTRACS_URL, out=str(output))
    print(f"\n[done] wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV path (e.g. $DATA_ROOT/ibtracs/ibTRACS_since_1980.csv).",
    )
    args = parser.parse_args()
    download_ibtracs(args.output)


if __name__ == "__main__":
    main()
