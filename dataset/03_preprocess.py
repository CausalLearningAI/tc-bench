from pathlib import Path
import logging
import numpy as np
import pandas as pd
import xarray as xr
import requests
from bs4 import BeautifulSoup
import wget
import json
from tqdm import tqdm
import zarr
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import math
import re
import os

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.since1980.list.v04r01.csv"
)
AGENCIES=['atcf', 'bom', 'hurdat_atl', 'hurdat_epa', 'nadi',
       'newdelhi', 'reunion', 'tokyo', 'wellington']
CAPITALS=['USA','BOM','USA','USA','NADI','NEWDELHI','REUNION','TOKYO','WELLINGTON']
AGENCY_INDEX=2  #example is KATRINA, so agency is hurdat_atl
VALID_TIMES = [f"{h:02d}:00:00" for h in range(0, 24, 3)]
HALF_WIDTH_POINTS = 112

GRIDSAT_URL = "https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/access/"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def download_ibtracs(url: str, target: Path) -> None:
    """
    Download IBTrACS CSV if it does not already exist.
    """
    if target.exists():
        logger.info("IBTrACS already exists: %s", target)
        return

    logger.info("Downloading IBTrACS to %s", target)
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        wget.download(url, out=str(target))
        logger.info("Download completed")
    except Exception as e:
        logger.error("Failed to download IBTrACS: %s", e)
        raise


def load_ibtracs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    first_row = df.iloc[0].astype(str).str.strip()
    df.columns = [
        f"{col} {unit}".strip() if "degrees" not in unit else col
        for col, unit in zip(df.columns, first_row)
    ]

    df = df.iloc[1:].reset_index(drop=True)
    
    return df


def enforce_3h_resolution(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["ISO_TIME"].str[-8:].isin(VALID_TIMES)
    df = df.loc[mask].copy()

    iso_dt = pd.to_datetime(df["ISO_TIME"])
    df["time_gap"] = (
    iso_dt.groupby(df["SID"])
          .diff()
          .dt.total_seconds() / 3600)
    
    return df
    
def clean_and_interpolate_track(
    track: pd.DataFrame,
    cols=("WMO_WIND kts", "WMO_PRES mb"),
) -> pd.DataFrame | None:
    
    """
    Trim leading/trailing NaNs in wind/pressure and then linearly interpolate .

    Returns
    -------
    pd.DataFrame or None
        Cleaned track, or None if no valid data exist.
    """
    track = track.sort_values("ISO_TIME")
    valid = track[list(cols)].notna().any(axis=1)
    if not valid.any():
        return None

    first = valid.idxmax()
    last = valid[::-1].idxmax()
    
    track = track.loc[first:last].copy()
    track[list(cols)] = track[list(cols)].interpolate(method='linear')

    return track
    
def split_by_agency(df: pd.DataFrame, agency, capital):

    """
    Splits the whole dataset according to WMO agency, for agency-sensitive analyses
    Returns:
    pd dataframe
    """
    wind_cols=[]
    lat_lon=[]
    lat_lon=list(df.columns[np.logical_or(df.columns.str.contains('LAT'),df.columns.str.contains('LON'))])
    wind_press=list(df.columns[np.logical_or(df.columns.str.contains('WIND'),df.columns.str.contains('PRES'))])
            
    for col in wind_press+lat_lon:
        df[col]=pd.to_numeric(df[col],errors='coerce')

        
    df["WMO_AGENCY"] = df["WMO_AGENCY"].astype(str).str.strip().replace("", np.nan)
    agency_per_sid = (
        df.groupby("SID")["WMO_AGENCY"]
        .apply(lambda x: x.dropna().unique())
    )
    valid_sid = agency_per_sid[agency_per_sid.map(len) == 1]
    sid_to_agency = valid_sid.map(lambda x: x[0])
    df = df[df["SID"].isin(sid_to_agency.index)].copy()
    df["WMO_AGENCY"] = df["SID"].map(sid_to_agency)

   
    logger.info("Processing agency: %s", agency)

    sub = df[df["WMO_AGENCY"] == agency]
    cols = (
            ["SID", "SEASON Year", "NUMBER", "BASIN", "SUBBASIN",
             "NAME", "ISO_TIME", "NATURE", "LAT", "LON", "time_gap"]
            + list(sub.columns[sub.columns.str.contains(capital)])
            + list(sub.columns[sub.columns.str.contains("WMO")]))
    sub = sub[cols]

    sub = (sub.groupby("SID", group_keys=False).apply(clean_and_interpolate_track).dropna(how="all"))
    sub=sub.reset_index(drop=True)
    return sub
        # out_file = out_dir / f"dataset_ibtracs_basic_cols_{agency}.csv"
        # sub.to_csv(out_file, index=False)

def list_gridsat_times(df):
    
    times=df['ISO_TIME']
    times=times.str.replace('-','.')
    times=times.str.split(':').str[0]
    times=times.str.replace(' ','.')
    
    return times

    
def list_gridsat_files(year: str, times: list[str]) -> list[str]:
    url = f"{GRIDSAT_URL}/{year}/"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".nc") and any(time in href for time in times):
            # make full URL if needed
            full_link = href if href.startswith("http") else url + href
            links.append(full_link)

    return links



def download_gridsat_files(
    year: str,
    times: list[str],
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    links = list_gridsat_files(year,times)

    for link in links:
        fname = link.split("/")[-1]
        target = out_dir / fname
        
        if target.exists():
            continue

        if not any(t in link for t in times):
            continue

        try:
            logger.info("Downloading %s", fname)
            wget.download(link, out=str(out_dir))
        except Exception as e:
            logger.error("Failed downloading %s: %s", fname, e)


def extract_patch(
    ds: xr.Dataset,
    lat_c: float,
    lon_c: float,
    half_width: int,
) -> xr.Dataset:
    lat = ds.lat.values
    lon = ds.lon.values

    i_lat = np.argmin(np.abs(lat - lat_c))
    i_lon = np.argmin(np.abs(lon - lon_c))

    lon_idx = (np.arange(i_lon - half_width, i_lon + half_width) % len(lon))
    lat_idx = np.arange(i_lat - half_width, i_lat + half_width)

    return ds.isel(
        lon=xr.DataArray(lon_idx, dims="lon"),
        lat=xr.DataArray(lat_idx, dims="lat"),
    )
    
def fill_nan_pixels_xr(
    frame_da: xr.DataArray,
    max_nan_distance_threshold: float = 0.02,
) -> xr.DataArray | None:
    """
    Fill NaN pixels in a 2D xarray DataArray using spatial nearest-neighbor interpolation.

    Parameters
    ----------
    frame_da : xr.DataArray
        2D DataArray (lat, lon)
    max_nan_distance_threshold : float
        Maximum allowed NaN distance as fraction of frame width

    Returns
    -------
    xr.DataArray or None
        Filled frame, or None if frame is invalid
    """
    frame = frame_da.values

    if not np.isnan(frame).any():
        return frame_da

    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    h, w = frame.shape
    y, x = np.mgrid[0:h, 0:w]

    valid_mask = ~np.isnan(frame)

    # Entire frame invalid
    if not valid_mask.any():
        return None

    valid_points = np.column_stack([y[valid_mask], x[valid_mask]])
    valid_values = frame[valid_mask]
    nan_points = np.column_stack([y[~valid_mask], x[~valid_mask]])

    tree = cKDTree(valid_points)
    distances, _ = tree.query(nan_points, k=1)

    max_distance = distances.max()
    distance_threshold = max_nan_distance_threshold * w

    if max_distance > distance_threshold:
        return None

    filled_values = griddata(
        valid_points,
        valid_values,
        nan_points,
        method="nearest",
    )

    filled = frame.copy()
    filled[~valid_mask] = filled_values

    filled = np.clip(filled, 140, 375)

    return xr.DataArray(
        filled,
        coords=frame_da.coords,
        dims=frame_da.dims,
        attrs=frame_da.attrs,
    ) # TODO: read this range from the metadata  # Ensure no negative values

    
def find_valid_frames_xr(
    frames_da: xr.DataArray,
    max_nan_distance_threshold: float = 0.02,
) -> xr.DataArray:
    """
    Determine which time frames are valid based on NaN distance criterion.
    """
    from scipy.spatial import cKDTree

    frames = frames_da.values
    t, h, w = frames.shape

    valid = np.ones(t, dtype=bool)
    distance_threshold = max_nan_distance_threshold * w

    y, x = np.mgrid[0:h, 0:w]

    for i in range(t):
        frame = frames[i]

        if not np.isnan(frame).any():
            continue

        valid_mask = ~np.isnan(frame)

        if not valid_mask.any():
            valid[i] = False
            continue

        valid_points = np.column_stack([y[valid_mask], x[valid_mask]])
        nan_points = np.column_stack([y[~valid_mask], x[~valid_mask]])

        tree = cKDTree(valid_points)
        distances, _ = tree.query(nan_points, k=1)

        if distances.max() > distance_threshold:
            valid[i] = False

    return xr.DataArray(valid, coords={"time": frames_da.time}, dims="time")


    

def rename_unnamed_cyclones(df: pd.DataFrame,agency) -> pd.DataFrame:


    """
    Unnecessary if the code is run for KATRINA only
    """
    
    unnamed = df[df["NAME"] == "UNNAMED"].copy()
    named = df[df["NAME"] != "UNNAMED"].copy()

    for i, sid in enumerate(unnamed["SID"].unique()):
        unnamed.loc[unnamed["SID"] == sid, "NAME"] = f"UNNAMED_{i}_{agency}"

    return pd.concat([named, unnamed], ignore_index=True)






    
def process_single_cyclone(
    df: pd.DataFrame,
    name: str,
    year: str,
    gridsat_dir: Path,
    out_dir: Path,
    var_press: str,
    var_wind: str,
):
    if os.path.exists(out_dir / f"{year}_{name}.nc")==True:
        logger.info("already existing %s", f"{year}_{name}")
        return
    cyclone = df[df["NAME"] == name].reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    patches = []
    missing_files = []
    skip_track=False
    times=list_gridsat_times(cyclone)
    for (_, row), time in zip(cyclone.iterrows(), times):

        fname = f"GRIDSAT-B1.{time}.v02r01.nc"
        path = gridsat_dir / year / fname

        if not path.exists():
            missing_files.append(time_str)
            continue

        try:
            ds = xr.open_dataset(path)[["irwin_cdr"]]

            patch = extract_patch(
                ds,
                lat_c=row["LAT"],
                lon_c=row["LON"],
                half_width=HALF_WIDTH_POINTS,
            )
            lat_ds=ds.lat.values
            lon_ds=ds.lon.values
            grid_spacing_lat=np.nanmean(np.diff(lat_ds))
            grid_spacing_lon=np.nanmean(np.diff(lon_ds))
            patch = patch.assign_coords(
                lat=np.linspace(-HALF_WIDTH_POINTS, HALF_WIDTH_POINTS - 1, HALF_WIDTH_POINTS*2),
                lon=np.linspace(-HALF_WIDTH_POINTS, HALF_WIDTH_POINTS - 1, HALF_WIDTH_POINTS*2),
            )

            patch["LAT_center"] = row["LAT"]
            patch["LON_center"] = row["LON"]
            patch["Min_pressure_mb"] = row[var_press]
            patch["Max_wind_kts"] = row[var_wind]


            ir = patch["irwin_cdr"].isel(time=0)

            filled = fill_nan_pixels_xr(ir,max_nan_distance_threshold=0.02)

            if filled is None:
                logger.warning("Dropping frame %s (%s): invalid NaN pattern", name, time_str)
                skip_track=True
                break 
                
            else:
                patch["irwin_cdr"][0] = filled
                patches.append(patch)


        except Exception as e:
            logger.error("Failed %s %s: %s", name, time_str, e)

    if skip_track==True:
        logger.info("Discarded %s", f"{year}_{name}"," - corrupted frames")
        return

    else:
        cyclone_ds = xr.concat(patches, dim="time")
        cyclone_ds.attrs["missing_times"] = missing_files
        out_file = out_dir / f"{year}_{name}.nc"
        
        cyclone_ds.to_netcdf(out_file)
        logger.info("Saved %s", out_file)



def process_year(
    df: pd.DataFrame,
    agency: str,
    agency_capital: str,
    year: int,
    gridsat_dir: Path,
    out_dir: Path,
):
    year = str(year)
    df_year = df[df["ISO_TIME"].str.contains(year)].copy()

    if df_year.empty:
        return

    df_year = rename_unnamed_cyclones(df_year,agency)

    var_press = f"{agency_capital}_PRES mb"
    var_wind = f"{agency_capital}_WIND kts"

    for col in [var_press, var_wind]:
        df_year[col] = pd.to_numeric(df_year[col], errors="coerce")

    names = df_year["NAME"].unique()

    for name in names:
        logger.info("Processing %s (%s)", name, year)
        process_single_cyclone(
            df=df_year,
            name=name,
            year=year,
            gridsat_dir=gridsat_dir,
            out_dir=out_dir,
            var_press=var_press,
            var_wind=var_wind,
        )


def main():
    agency = AGENCIES[AGENCY_INDEX]
    agency_capital = CAPITALS[AGENCY_INDEX]
    
    base_dir = Path(".")
    data_dir = base_dir / "data"
    gridsat_dir = base_dir / "gridsat"
    output_dir = base_dir / "cyclone_windows"/agency
    ibtracs_csv = data_dir / "ibtracs_since_1980.csv"
    
    download_ibtracs(url=IBTRACS_URL,target=ibtracs_csv)
    year=YEAR    
    logger.info("Loading IBTrACS")
    df = load_ibtracs(ibtracs_csv)
    df = enforce_3h_resolution(df)
    
    

    if os.path.exists( data_dir / f"dataset_ibtracs_basic_cols_{agency}.csv")==True:
        
        df_agency = pd.read_csv(data_dir / f"dataset_ibtracs_basic_cols_{agency}.csv")
    else:
        df_agency=split_by_agency(df,agency,agency_capital)
        df_agency.to_csv(data_dir / f"dataset_ibtracs_basic_cols_{agency}.csv")
        
    df_agency["LON"] = (df_agency["LON"] + 180) % 360 - 180
    df_year= df_agency.loc[df_agency["ISO_TIME"].str.contains(YEAR)].reset_index(drop=True)
    
    if ONE_CYCLONE_ONLY==True:
        df_name=df_year.loc[df_year['NAME'].str.contains(EXAMPLE_CYCLONE)].reset_index(drop=True)
    else:
        df_name=df_year
    if len(df_name)==0:
        logger.info("empty dataset %s %s (%s)", agency, name, year)
        exit(1)
    times=list_gridsat_times(df_name).values
    
    download_gridsat_files(
        year=str(year),
        times=times,
        out_dir=gridsat_dir / str(year),
    )

    process_year(
        df=df_name,
        agency=agency,
        agency_capital=agency_capital,
        year=year,
        gridsat_dir=gridsat_dir,
        out_dir=output_dir,
    )
    
if __name__ == "__main__":
    main()    



    