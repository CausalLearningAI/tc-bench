import xarray as xr
import numpy as np
from pathlib import Path
import json
import os
from tqdm import tqdm
import zarr
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import math
import re
import pandas as pd
from scipy.ndimage import label

FILL_VALUE = 200


PRESSURE_KEY = 'Min pressure mb'
WIND_KEY = 'Max wind kts'
LAT_KEY = 'LAT center'
LON_KEY = 'LON center'
FRAME_KEY = 'irwin_cdr'

class CycloneDataConsolidator:
    """
    Consolidate individual .nc trajectory files into efficient formats.

    Expected input structure:
        base_path/$agency/$name_$year.nc

    Each .nc file contains a complete trajectory with dimensions (time, lat, lon).
    Supports optional NaN handling and variable-length sequences.
    Temporal interpolation is available but disabled by default.
    """

    def __init__(self,
                 base_path: str | None = None,
                 out_path: str | None = None,
                 min_frames: int = 1,
                 debug: bool = False,
                 workers: int = 1):
        _data_root = os.environ.get("DATA_ROOT", os.path.join(os.environ["HOME"], "tcbench"))
        if base_path is None:
            base_path = os.environ.get("CROPPED_DIR", os.path.join(_data_root, "cropped"))
        if out_path is None:
            out_path = os.environ.get("CONSOLIDATED_DIR", os.path.join(_data_root, "consolidated"))
        self.base_path = Path(base_path)
        self.output_base = Path(out_path)
        self.min_frames = min_frames
        self.debug = debug
        self.workers = workers
    
    def interpolate_missing_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """
        Interpolate missing (NaN) coordinate values using linear interpolation/extrapolation.
        
        Parameters:
        -----------
        coords : np.ndarray
            Coordinate array (may contain NaNs)
        
        Returns:
        --------
        filled_coords : np.ndarray
            Array with NaNs interpolated
        """
        if not np.any(np.isnan(coords)):
            # No NaNs, return as is
            return coords
        
        # Get valid (non-NaN) indices and values
        valid_mask = ~np.isnan(coords)
        valid_indices = np.where(valid_mask)[0]
        valid_values = coords[valid_mask]
        
        if len(valid_values) == 0:
            # All NaN - cannot interpolate, return as is
            return coords
        
        if len(valid_values) == 1:
            # Only one valid value - use it for all
            return np.full_like(coords, valid_values[0])
        
        # Interpolate and extrapolate
        all_indices = np.arange(len(coords))
        interp_func = interp1d(valid_indices, valid_values, 
                              kind='linear', 
                              bounds_error=False, 
                              fill_value='extrapolate')
        
        return interp_func(all_indices)

    def find_valid_frames(
        self,
        frames: np.ndarray,
        *,
        min_valid_frac: float = 0.995,
        center_radius_frac: float = 0.18,
        seam_frac: float = 0.8,
        max_nan_cc_frac: float = 0.01,
        debug: bool = False,
    ) -> np.ndarray:
        """
        Structure-aware QC for centered cyclone crops.

        Frames are invalid if:
        - too many NaNs overall
        - NaNs near the image center (cyclone core)
        - seam / scanline-like NaN structures
        - large connected NaN components
        """

        T, H, W = frames.shape
        assert H == W, "Expected square crops"

        cx = cy = W // 2
        center_radius = int(center_radius_frac * W)

        yy, xx = np.ogrid[:H, :W]
        center_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= center_radius ** 2

        valid_mask = np.ones(T, dtype=bool)

        for t in range(T):
            frame = frames[t]
            finite = np.isfinite(frame)
            nan = ~finite

            # 1) Global NaN fraction
            valid_frac = finite.mean()
            if valid_frac < min_valid_frac:
                valid_mask[t] = False
                if debug:
                    print(f"[frame {t}] invalid: valid_frac={valid_frac:.3f}")
                continue

            # 2) NaNs near cyclone center
            if (nan & center_mask).any():
                valid_mask[t] = False
                if debug:
                    print(f"[frame {t}] invalid: NaNs near center")
                continue

            # 3) Seam / scanline detection
            col_nan_frac = nan.mean(axis=0)
            row_nan_frac = nan.mean(axis=1)
            if (col_nan_frac > seam_frac).any() or (row_nan_frac > seam_frac).any():
                valid_mask[t] = False
                if debug:
                    print(f"[frame {t}] invalid: seam-like NaN structure")
                continue

            # 4) Large connected NaN components
            labeled, num_cc = label(nan)
            if num_cc > 0:
                total_pixels = H * W
                for cc_id in range(1, num_cc + 1):
                    cc_size = np.sum(labeled == cc_id)
                    if cc_size / total_pixels > max_nan_cc_frac:
                        valid_mask[t] = False
                        if debug:
                            print(f"[frame {t}] invalid: large NaN component ({cc_size}px)")
                        break

        return valid_mask

    
    def check_delta_timestamps(self, timestamps: list, expected_delta: timedelta = timedelta(hours=3)) -> list:
        """
        Check for gaps in timestamps based on expected delta.

        Parameters:
        -----------
        timestamps : list
            List of datetime objects
        expected_delta : timedelta
            Expected time difference between consecutive timestamps

        Returns:
        --------
        gap_info : list
            List of tuples (index_before_gap, index_after_gap, actual_delta)
        """
        for i in range(1, len(timestamps)):
            time_diff = timestamps[i] - timestamps[i-1]
            time_diff_in_timedelta = np.timedelta64(time_diff, 'ns').astype('timedelta64[s]').item()
            if time_diff_in_timedelta != expected_delta:
                return False
        return True
            

    def handle_pressure_wind_nans(self, pressure_vals: np.ndarray, wind_vals: np.ndarray,
                                  strategy: str = "trim_ends_interpolate_middle",
                                  frame_valid_mask: np.ndarray = None) -> tuple:
        """
        Handle NaNs in pressure and wind data, optionally considering frame validity.

        Strategies:
        -----------
        1. "trim_ends_interpolate_middle":
           - Remove leading/trailing NaNs (and invalid frames if frame_valid_mask provided)
           - Takes intersection of valid regions for pressure, wind, and frames
           - Interpolate in-between NaNs

        2. "forward_fill":
           - Carry last valid value forward
           - Good for simple missing data

        3. "interpolate_all":
           - Linear interpolation for all NaNs
           - Assumes gradual changes

        4. "keep_nans":
           - Keep NaNs and return validity mask
           - For uncertainty estimation

        Parameters:
        -----------
        pressure_vals : np.ndarray
            Pressure values array
        wind_vals : np.ndarray
            Wind values array
        strategy : str
            NaN handling strategy
        frame_valid_mask : np.ndarray, optional
            Boolean mask indicating valid frames (True=valid, False=invalid)

        Returns:
        --------
        pressure_clean : np.ndarray
            Cleaned pressure array
        wind_clean : np.ndarray
            Cleaned wind array
        validity_mask : np.ndarray or None
            1 for valid, 0 for filled/interpolated NaNs
        trimmed_indices : tuple
            (start_idx, end_idx) of valid region
        """
        
        def find_valid_region(arr):
            """Find first and last non-NaN indices."""
            valid_mask = ~np.isnan(arr)
            if not valid_mask.any():
                return None, None
            
            valid_indices = np.where(valid_mask)[0]
            return valid_indices[0], valid_indices[-1]
        
        def trim_ends(arr):
            """Remove leading and trailing NaNs."""
            start_idx, end_idx = find_valid_region(arr)
            if start_idx is None:
                return arr, 0, len(arr)
            return arr[start_idx:end_idx+1], start_idx, end_idx+1
        
        def interpolate_nans(arr):
            """Linearly interpolate NaNs."""
            x = np.arange(len(arr))
            valid_mask = ~np.isnan(arr)
            
            if not valid_mask.any():
                return arr
            
            if valid_mask.all():
                return arr
            
            # Interpolate
            arr_interp = np.interp(x, x[valid_mask], arr[valid_mask])
            return arr_interp
        
        def forward_fill_nans(arr):
            """Forward fill NaNs with last valid value."""
            arr_filled = arr.copy()
            mask = np.isnan(arr_filled)
            idx = np.where(~mask, np.arange(len(mask)), 0)
            idx = np.maximum.accumulate(idx)
            arr_filled[mask] = arr_filled[idx[mask]]
            return arr_filled
        
        # Record original NaN positions for mask
        original_nan_mask = np.isnan(pressure_vals) | np.isnan(wind_vals)
        
        if strategy == "trim_ends_interpolate_middle":
            # Case 1: Find valid region (trim ends) for pressure and wind
            p_start, p_end = find_valid_region(pressure_vals)
            w_start, w_end = find_valid_region(wind_vals)

            if p_start is None or w_start is None:
                if p_start is None and w_start is None:
                    print("  NaN handling failed: Both pressure and wind are ALL NaN")
                elif p_start is None:
                    print("  NaN handling failed: Pressure is ALL NaN")
                else:
                    print("  NaN handling failed: Wind is ALL NaN")
                return None  # Cannot recover from all NaNs

            # Find valid region for frames if frame_valid_mask is provided
            if frame_valid_mask is not None:
                if not np.any(frame_valid_mask):
                    print("  NaN handling failed: All frames are invalid (NaN pixels too far from valid pixels)")
                    return None

                # Find first and last valid frame indices
                valid_frame_indices = np.where(frame_valid_mask)[0]
                f_start = valid_frame_indices[0]
                f_end = valid_frame_indices[-1]

                # Use intersection of valid regions for pressure, wind, AND frames
                start_idx = max(p_start, w_start, f_start)
                end_idx = min(p_end, w_end, f_end)

                if start_idx > end_idx:
                    print(f"  NaN handling failed: No overlapping valid region "
                          f"(pressure valid: [{p_start}:{p_end+1}], "
                          f"wind valid: [{w_start}:{w_end+1}], "
                          f"frames valid: [{f_start}:{f_end+1}])")
                    return None
            else:
                # Use intersection of valid regions for pressure and wind only
                start_idx = max(p_start, w_start)
                end_idx = min(p_end, w_end)
                f_start, f_end = None, None

                if start_idx > end_idx:
                    print(f"  NaN handling failed: No overlapping valid region "
                          f"(pressure valid: [{p_start}:{p_end+1}], "
                          f"wind valid: [{w_start}:{w_end+1}])")
                    return None

            # Trim both arrays
            pressure_clean = pressure_vals[start_idx:end_idx+1]
            wind_clean = wind_vals[start_idx:end_idx+1]

            # Count in-between NaNs before interpolation
            n_inbetween_p = np.sum(np.isnan(pressure_clean))
            n_inbetween_w = np.sum(np.isnan(wind_clean))

            # Create validity mask BEFORE interpolation (tracks which values were originally valid)
            validity_mask = ~np.isnan(pressure_clean) & ~np.isnan(wind_clean)

            # Interpolate in-between NaNs for pressure and wind
            pressure_clean = interpolate_nans(pressure_clean)
            wind_clean = interpolate_nans(wind_clean)

            n_trimmed_start = start_idx
            n_trimmed_end = len(pressure_vals) - (end_idx + 1)

            # Build info message
            if frame_valid_mask is not None:
                n_invalid_frames = np.sum(~frame_valid_mask)
                print(f"  NaN handling (trim_ends_interpolate_middle): "
                      f"Removed {n_trimmed_start} leading + {n_trimmed_end} trailing frames, "
                      f"Kept frames [{start_idx}:{end_idx+1}], "
                      f"Interpolated {n_inbetween_p} pressure + {n_inbetween_w} wind in-between NaNs, "
                      f"{n_invalid_frames} invalid frames detected")
            else:
                print(f"  NaN handling (trim_ends_interpolate_middle): "
                      f"Removed {n_trimmed_start} leading + {n_trimmed_end} trailing frames, "
                      f"Kept frames [{start_idx}:{end_idx+1}], "
                      f"Interpolated {n_inbetween_p} pressure + {n_inbetween_w} wind in-between NaNs")

            return pressure_clean, wind_clean, validity_mask.astype(np.int8), (start_idx, end_idx+1)
        
        elif strategy == "forward_fill":
            pressure_clean = forward_fill_nans(pressure_vals)
            wind_clean = forward_fill_nans(wind_vals)
            
            validity_mask = ~original_nan_mask
            
            print(f"  NaN handling (forward_fill): Filled {np.sum(original_nan_mask)} NaNs")
            return pressure_clean, wind_clean, validity_mask.astype(np.int8), (0, len(pressure_vals))
        
        elif strategy == "interpolate_all":
            pressure_clean = interpolate_nans(pressure_vals)
            wind_clean = interpolate_nans(wind_vals)
            
            validity_mask = ~original_nan_mask
            
            print(f"  NaN handling (interpolate_all): Interpolated {np.sum(original_nan_mask)} NaNs")
            return pressure_clean, wind_clean, validity_mask.astype(np.int8), (0, len(pressure_vals))
        
        elif strategy == "keep_nans":
            # Keep NaNs and return mask for downstream handling
            validity_mask = ~original_nan_mask
            
            print(f"  NaN handling (keep_nans): Kept {np.sum(original_nan_mask)} NaNs, created validity mask")
            return pressure_vals, wind_vals, validity_mask.astype(np.int8), (0, len(pressure_vals))
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def consolidate_cyclone_to_netcdf(self, location: str, year: int, name: str,
                                     pressure_wind_strategy: str = "trim_ends_interpolate_middle",
                                     dry_run: bool = False):
        """
        Consolidate a single trajectory .nc file with interpolation and optional padding.

        Input file structure: base_path/$agency/$year_$name.nc
        Each .nc file contains a complete trajectory with dimensions (time, lat, lon).

        Parameters:
        -----------
        location : str
            Agency/location name (subdirectory under base_path)
        year : int
            Year of the cyclone
        name : str
            Name of the cyclone
        pressure_wind_strategy : str
            How to handle NaNs in pressure/wind data
        dry_run : bool
            If True, only check if cyclone would be valid without saving
        """
        # Check if output file already exists (skip in dry_run mode)
        output_dir = self.output_base / location
        output_path = output_dir / f"{year}_{name}.nc"

        input_path = self.base_path / location / f"{year}_{name}.nc"

        if not input_path.exists():
            print(f"  SKIP REASON: File not found: {input_path}")
            return None

        # Load the trajectory file
        try:
            ds = xr.open_dataset(input_path)
        except Exception as e:
            print(f"  SKIP REASON: Failed to open {input_path}: {e}")
            return None

        frames_array = ds[FRAME_KEY].values  # Expecting shape (time, lat, lon)
        pressure_array = ds[PRESSURE_KEY].values
        wind_array = ds[WIND_KEY].values
        lat_center_array = ds[LAT_KEY].values
        lon_center_array = ds[LON_KEY].values
        n_frames = frames_array.shape[0]

        # Extract timestamps from time coordinate
        timestamps = ds.coords['time'].values
        timestamps = pd.to_datetime(timestamps, format="%Y.%m.%d.%H").to_numpy()
        
        # Data Range
        valid_range = ds[FRAME_KEY].attrs.get('valid_range', None)
        frame_min, frame_max = valid_range[0], valid_range[1]
        ds.close()
        
        # if timestamp does not follow strict dt, skip the trajectory
        if not self.check_delta_timestamps(timestamps):
            print(f"  SKIP REASON: Timestamps in {input_path} do not follow strict 3-hour intervals")
            return None

        if not (len(pressure_array) == len(wind_array) == len(frames_array) == n_frames):
            print(f"  SKIP REASON: Mismatched lengths in {input_path}")
            return None

        if n_frames == 0:
            print(f"  SKIP REASON: No frames in {input_path}")
            return None

        # Convert to numpy arrays
        cyclone_id = f"{location}_{year}_{name}"
        
        # Check if all values are NaN - skip this cyclone
        if np.all(np.isnan(pressure_array)) and np.all(np.isnan(wind_array)):
            print(f"{cyclone_id} SKIP REASON: All pressure and wind values are NaN")
            return None

        # =====================================================================
        # STEP 1: Find valid frames based on max NaN distance threshold (2% of frame width)
        # =====================================================================
        frame_valid_mask = self.find_valid_frames(frames_array)

        # =====================================================================
        # STEP 2: Trim ends and find intersection of valid regions
        # (pressure valid, wind valid, frames valid)
        # =====================================================================
        result = self.handle_pressure_wind_nans(
            pressure_array, wind_array,
            strategy=pressure_wind_strategy,
            frame_valid_mask=frame_valid_mask
        )

        pressure_array, wind_array, pres_wind_validity_mask, (trim_start, trim_end) = result

        # Trim all arrays to match valid region
        frames_array = frames_array[trim_start:trim_end]
        frame_valid_mask = frame_valid_mask[trim_start:trim_end]  # Also trim the validity mask
        timestamps = timestamps[trim_start:trim_end]
        lat_center_array = lat_center_array[trim_start:trim_end]
        lon_center_array = lon_center_array[trim_start:trim_end]
        # =====================================================================
        # STEP 2b: Check for invalid frames IN-BETWEEN after trimming
        # If any invalid frames remain in the middle, drop the entire trajectory
        # (We do NOT interpolate frames - only trim ends)
        # =====================================================================
        if not np.all(frame_valid_mask):
            invalid_indices = np.where(~frame_valid_mask)[0].tolist()
            print(f"{cyclone_id} SKIP REASON: Invalid frames in-between after trimming at indices {invalid_indices} - dropping trajectory")
            return None

        # =====================================================================
        # STEP 3: Check center coordinates for NaNs and report statistics
        # =====================================================================
        n_lat_nan = np.sum(np.isnan(lat_center_array))
        n_lon_nan = np.sum(np.isnan(lon_center_array))
        n_total = len(lat_center_array)

        if n_lat_nan > 0 or n_lon_nan > 0:
            lat_nan_indices = np.where(np.isnan(lat_center_array))[0].tolist()
            lon_nan_indices = np.where(np.isnan(lon_center_array))[0].tolist()
            print(f"  Center coordinate NaNs: {n_lat_nan}/{n_total} lat NaNs at {lat_nan_indices}, "
                  f"{n_lon_nan}/{n_total} lon NaNs at {lon_nan_indices}")

            # Interpolate missing coordinates (we allow this since it's just for visualization/tracking)
            lat_center_array = self.interpolate_missing_coordinates(lat_center_array)
            lon_center_array = self.interpolate_missing_coordinates(lon_center_array)

            # Check if interpolation succeeded
            if np.any(np.isnan(lat_center_array)) or np.any(np.isnan(lon_center_array)):
                print(f"  SKIP REASON: Could not interpolate all center coordinates - dropping trajectory")
                return None

        # =====================================================================
        # STEP 5: Fill NaN pixels in valid frames (small NaN regions only)
        # =====================================================================
        frames_array[frame_valid_mask][np.isnan(frames_array)] = FILL_VALUE

        # =====================================================================
        # STEP 6: Check minimum sequence length
        # =====================================================================
        
        # Create new consolidated dataset
        # Combine lat/lon into single center array (T, 2)
        center_array = np.stack([lat_center_array, lon_center_array], axis=1)  # (T, 2)
        frames_array = np.clip(frames_array, frame_min, frame_max) #TODO: clip to the valid range
        
        ds_consolidated = xr.Dataset(
            data_vars={
                FRAME_KEY: (["time", "y", "x"], frames_array),
                "pressure": (["time"], pressure_array),
                "wind": (["time"], wind_array),
                "frame_valid_mask": (["time"], frame_valid_mask.astype(np.int8)),
                "center": (["time", "coords"], center_array),
            },
            coords={
                "time": np.datetime_as_string(timestamps, unit='h').tolist(),
                "y": np.arange(frames_array.shape[1]),
                "x": np.arange(frames_array.shape[2]),
                "coords": ["lat", "lon"],
            },
            attrs={
                "cyclone_name": name,
                "year": year,
                "location": location,
                "num_frames": len(frames_array),
                "num_valid_frames": np.sum(frame_valid_mask),
                "original_variable_name": FRAME_KEY,
                "pressure_wind_strategy": pressure_wind_strategy,
                "pressure_units": "mb (millibars)",
                "wind_units": "kts (knots)",
                "center_units": "[degrees_north, degrees_east]",
                "center_description": "Cyclone center coordinates [lat, lon]",
            }
        )
        
        ds_consolidated[FRAME_KEY].attrs = ds[FRAME_KEY].attrs  # Copy frame attributes

        # Store validity mask for pressure/wind
        if pres_wind_validity_mask is not None:
            ds_consolidated["pressure_wind_valid"] = (["time"], pres_wind_validity_mask)
        
        # In dry_run mode, just return success indicator without saving
        if dry_run:
            ds_consolidated.close()
            return "dry_run_success"

        # Save consolidated file
        output_dir = self.output_base / location
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{year}_{name}.nc"

        # Use compression
        encoding = {
            FRAME_KEY: {"zlib": True, "complevel": 5, "dtype": "float32"},
            "pressure": {"zlib": True, "complevel": 5, "dtype": "float32"},
            "wind": {"zlib": True, "complevel": 5, "dtype": "float32"},
            "center": {"zlib": True, "complevel": 5, "dtype": "float32"},
        }

        try:
            ds_consolidated.to_netcdf(output_path, encoding=encoding)
            ds_consolidated.close()
            return output_path
        except Exception as e:
            print(f"  Error saving {output_path}: {e}")
            ds_consolidated.close()
            return None
    
    def process_cyclone_wrapper(self, args):
        """Wrapper for multiprocessing - unpacks args and calls consolidate_cyclone_to_netcdf."""
        location, year, name = args
        try:
            result = self.consolidate_cyclone_to_netcdf(location, year, name)
            if result:
                # Return success with path for statistics gathering
                return ('success', result, location, year, name)
            else:
                return ('skipped', None, location, year, name)
        except Exception as e:
            return ('failed', str(e), location, year, name)
    
    def consolidate_all(self, format: str = "netcdf", dry_run: bool = False):
        """
        Consolidate all cyclones.

        Parameters:
        -----------
        format : str
            Output format (only "netcdf" supported)
        dry_run : bool
            If True, only compute statistics without saving files
        """
        stats = {"success": 0, "failed": 0, "skipped": 0}
        sequence_lengths = []

        # Per-agency (location) statistics
        agency_stats = {}

        # Collect data for statistics only in debug mode
        if self.debug:
            all_pressure = []
            all_wind = []
            all_lat = []
            all_lon = []
            # Frame statistics computed incrementally to avoid memory issues
            frame_stats = {
                'count': 0,
                'sum': 0.0,
                'sum_sq': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'nan_count': 0
            }
        
        # First pass: count total cyclones for progress bar
        # New structure: base_path/$agency/$year_$name.nc
        total_cyclones = 0
        for location_dir in self.base_path.iterdir():
            if not location_dir.is_dir():
                continue
            for nc_file in location_dir.glob("*.nc"):
                # Parse filename: $year_$name.nc
                stem = nc_file.stem  # filename without .nc
                # Find the first underscore to split year and name
                first_underscore = stem.find('_')
                if first_underscore > 0:
                    try:
                        int(stem[:first_underscore])  # Validate year is numeric
                        total_cyclones += 1
                    except ValueError:
                        continue

        print(f"Found {total_cyclones} cyclones to process\n")

        # Collect all cyclone tasks
        # New structure: base_path/$agency/$year_$name.nc
        cyclone_tasks = []
        for location_dir in self.base_path.iterdir():
            if not location_dir.is_dir():
                continue
            location = location_dir.name

            for nc_file in location_dir.glob("*.nc"):
                # Parse filename: $name_$year.nc
                stem = nc_file.stem  # filename without .nc
                # Find the last underscore to split name and year
                first_underscore = stem.rfind('_')
                if first_underscore <= 0:
                    continue

                # Validate year is numeric
                try:
                    year = int(stem[:first_underscore])
                    name = stem[first_underscore+1:]
                    cyclone_tasks.append((location, year, name))
                except ValueError:
                    continue
        
        # Process cyclones with multiprocessing
        from tqdm import tqdm
        from multiprocessing import Pool
        
        pbar = tqdm(total=len(cyclone_tasks), desc="Consolidating", unit="cyclone")
        
        def update_agency_stats(agency_stats, location, status):
            """Helper to update per-agency statistics."""
            if location not in agency_stats:
                agency_stats[location] = {"total": 0, "success": 0, "skipped": 0, "failed": 0}
            agency_stats[location]["total"] += 1
            agency_stats[location][status] += 1

        assert self.workers > 1, "Workers must be greater than 1 for multiprocessing"
        # Multiprocessing mode
        with Pool(self.workers) as pool:
            for status, result, location, year, name in pool.imap_unordered(self.process_cyclone_wrapper, cyclone_tasks):
                if status == 'success':
                    stats["success"] += 1
                    update_agency_stats(agency_stats, location, 'success')

                    if not dry_run:
                        # Track sequence length
                        ds = xr.open_dataset(result)
                        sequence_lengths.append(ds.attrs['num_frames'])

                        if self.debug:
                            # Collect statistics only in debug mode
                            all_pressure.extend(ds["pressure"].values.tolist())
                            all_wind.extend(ds["wind"].values.tolist())

                            center_data = ds["center"].values
                            all_lat.extend(center_data[:, 0].tolist())
                            all_lon.extend(center_data[:, 1].tolist())

                            frames_data = ds[FRAME_KEY].values
                            frame_stats['count'] += frames_data.size
                            frame_stats['sum'] += np.sum(frames_data)
                            frame_stats['sum_sq'] += np.sum(frames_data ** 2)
                            frame_stats['min'] = min(frame_stats['min'], np.min(frames_data))
                            frame_stats['max'] = max(frame_stats['max'], np.max(frames_data))
                            frame_stats['nan_count'] += np.sum(np.isnan(frames_data))

                        ds.close()
                elif status == 'skipped':
                    stats["skipped"] += 1
                    update_agency_stats(agency_stats, location, 'skipped')
                else:  # failed
                    stats["failed"] += 1
                    update_agency_stats(agency_stats, location, 'failed')
                    if self.debug:
                        print(f"✗ FAILED: {location}/{year}_{name}: {result}")

                pbar.set_postfix({"success": stats["success"], "skipped": stats["skipped"], "failed": stats["failed"]})
                pbar.update(1)
        pbar.close()
        
        print("\n" + "="*80)
        if dry_run:
            print(f"DRY RUN complete (no files saved)!")
        else:
            print(f"Consolidation complete!")
        print(f"  Success: {stats['success']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Skipped: {stats['skipped']}")

        # Print per-agency statistics
        if agency_stats:
            print(f"\n" + "="*80)
            print(f"PER-AGENCY (LOCATION) STATISTICS")
            print(f"="*80)
            print(f"{'Agency':<20} {'Total':>8} {'Success':>8} {'Skipped':>8} {'Failed':>8} {'Success%':>10}")
            print("-"*80)
            for agency in sorted(agency_stats.keys()):
                s = agency_stats[agency]
                success_pct = 100.0 * s['success'] / s['total'] if s['total'] > 0 else 0
                print(f"{agency:<20} {s['total']:>8} {s['success']:>8} {s['skipped']:>8} {s['failed']:>8} {success_pct:>9.1f}%")
            print("-"*80)
            # Print total row
            total_total = sum(s['total'] for s in agency_stats.values())
            total_success = sum(s['success'] for s in agency_stats.values())
            total_skipped = sum(s['skipped'] for s in agency_stats.values())
            total_failed = sum(s['failed'] for s in agency_stats.values())
            total_pct = 100.0 * total_success / total_total if total_total > 0 else 0
            print(f"{'TOTAL':<20} {total_total:>8} {total_success:>8} {total_skipped:>8} {total_failed:>8} {total_pct:>9.1f}%")
            print(f"="*80)

        if self.debug and sequence_lengths:
            print(f"\nSequence length statistics:")
            print(f"  Min: {np.min(sequence_lengths)}")
            print(f"  Max: {np.max(sequence_lengths)}")
            print(f"  Mean: {np.mean(sequence_lengths):.1f}")
            print(f"  Median: {np.median(sequence_lengths):.1f}")
            print(f"  95th percentile: {np.percentile(sequence_lengths, 95):.1f}")
        
        # Compute and display statistics for pressure, wind, lat, lon
        if self.debug and all_pressure:
            pressure_array = np.array(all_pressure)
            wind_array = np.array(all_wind)
            lat_array = np.array(all_lat)
            lon_array = np.array(all_lon)
            
            # Compute frame statistics from accumulated values
            frame_mean = frame_stats['sum'] / frame_stats['count']
            frame_variance = (frame_stats['sum_sq'] / frame_stats['count']) - (frame_mean ** 2)
            frame_std = np.sqrt(max(0, frame_variance))  # Avoid negative due to numerical errors
            
            print(f"\n" + "="*80)
            print(f"DATA STATISTICS (all consolidated cyclones)")
            print(f"="*80)
            print(f"\nPressure (mb):")
            print(f"  Count: {len(pressure_array)}")
            print(f"  Mean: {np.mean(pressure_array):.2f}")
            print(f"  Std: {np.std(pressure_array):.2f}")
            print(f"  Min: {np.min(pressure_array):.2f}")
            print(f"  Max: {np.max(pressure_array):.2f}")
            print(f"  NaN count: {np.sum(np.isnan(pressure_array))}")
            
            print(f"\nWind (kts):")
            print(f"  Count: {len(wind_array)}")
            print(f"  Mean: {np.mean(wind_array):.2f}")
            print(f"  Std: {np.std(wind_array):.2f}")
            print(f"  Min: {np.min(wind_array):.2f}")
            print(f"  Max: {np.max(wind_array):.2f}")
            print(f"  NaN count: {np.sum(np.isnan(wind_array))}")
            
            print(f"\nLatitude (degrees):")
            print(f"  Count: {len(lat_array)}")
            print(f"  Mean: {np.mean(lat_array):.2f}")
            print(f"  Std: {np.std(lat_array):.2f}")
            print(f"  Min: {np.min(lat_array):.2f}")
            print(f"  Max: {np.max(lat_array):.2f}")
            print(f"  NaN count: {np.sum(np.isnan(lat_array))}")
            
            print(f"\nLongitude (degrees):")
            print(f"  Count: {len(lon_array)}")
            print(f"  Mean: {np.mean(lon_array):.2f}")
            print(f"  Std: {np.std(lon_array):.2f}")
            print(f"  Min: {np.min(lon_array):.2f}")
            print(f"  Max: {np.max(lon_array):.2f}")
            print(f"  NaN count: {np.sum(np.isnan(lon_array))}")
            
            print(f"\nFrames (satellite images):")
            print(f"  Total pixel count: {frame_stats['count']}")
            print(f"  Mean: {frame_mean:.2f}")
            print(f"  Std: {frame_std:.2f}")
            print(f"  Min: {frame_stats['min']:.2f}")
            print(f"  Max: {frame_stats['max']:.2f}")
            print(f"  NaN count: {frame_stats['nan_count']}")
            print(f"="*80)
        
        return stats, sequence_lengths, agency_stats


# def load_consolidated_cyclone(file_path: str):
#     """Example: Load a consolidated cyclone file with pressure and wind."""
#     ds = xr.open_dataset(file_path)
    
#     print(f"Cyclone: {ds.attrs['cyclone_name']} ({ds.attrs['year']})")
#     print(f"Shape: {ds[ds.attrs['original_variable']].shape}")
#     print(f"Frames: {ds.attrs['num_frames']}")
    
#     # Access all frames at once
#     all_frames = ds[ds.attrs['original_variable']].values  # (T, H, W)
    
#     # Access pressure and wind
#     pressure = ds["pressure"].values  # (T,)
#     wind = ds["wind"].values  # (T,)
    
#     print(f"\nPressure range: [{np.nanmin(pressure):.1f}, {np.nanmax(pressure):.1f}] mb")
#     print(f"Wind range: [{np.nanmin(wind):.1f}, {np.nanmax(wind):.1f}] kts")
    
#     # Access specific frame with its pressure/wind
#     if len(all_frames) > 10:
#         frame_10 = ds[ds.attrs['original_variable']].isel(time=10).values
#         pressure_10 = ds["pressure"].isel(time=10).values.item()
#         wind_10 = ds["wind"].isel(time=10).values.item()
        
#         print(f"\nFrame 10:")
#         print(f"  Pressure: {pressure_10:.1f} mb")
#         print(f"  Wind: {wind_10:.1f} kts")
        
#         # Access metadata for specific frame
#         metadata_str = ds["metadata"].isel(time=10).values.item()
#         metadata_10 = json.loads(metadata_str)
#         print(f"  Full metadata: {metadata_10}")
    
#     return ds


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    debug_mode = "--debug" in sys.argv
    test_mode = "--test" in sys.argv
    dry_run_mode = "--dry-run" in sys.argv

    # Parse --workers argument
    workers = 1
    for i, arg in enumerate(sys.argv):
        if arg == "--workers" and i + 1 < len(sys.argv):
            try:
                workers = int(sys.argv[i + 1])
                if workers < 1:
                    workers = 1
                    print("Warning: workers must be >= 1, using 1")
            except ValueError:
                print(f"Warning: invalid workers value '{sys.argv[i + 1]}', using 1")
                workers = 1
            break

    _data_root = os.environ.get("DATA_ROOT", os.path.join(os.environ["HOME"], "tcbench"))
    if test_mode:
        base_path = os.environ.get("CROPPED_TEST_DIR", os.path.join(_data_root, "test"))
        out_path = os.environ.get("CONSOLIDATED_TEST_DIR", os.path.join(_data_root, "consolidated_test"))
        print("="*80)
        print("TESTING MODE: Consolidating test cyclones with NaN handling")
        print("="*80)
    else:
        base_path = os.environ.get("CROPPED_DIR", os.path.join(_data_root, "cropped"))
        out_path = os.environ.get("CONSOLIDATED_DIR", os.path.join(_data_root, "consolidated"))
        print("="*80)
        if dry_run_mode:
            print("DRY RUN: Checking cyclone validity without saving files")
        else:
            print("Consolidating all cyclones with NaN handling")
        print("="*80)

    if debug_mode:
        print("Debug mode: ON (verbose output and statistics enabled)")
    if dry_run_mode:
        print("Dry run mode: ON (statistics only, no files saved)")
    if workers > 1:
        print(f"Workers: {workers} (parallel processing)")

    consolidator = CycloneDataConsolidator(
        base_path=base_path,
        out_path=out_path,
        debug=debug_mode,
        workers=workers
    )

    # Using "trim_ends_interpolate_middle" strategy:
    # - Removes leading/trailing NaNs (weak formation/dissipation phases)
    # - Interpolates in-between NaNs

    print("\nStarting consolidation of all cyclones...")
    print(f"Source: {consolidator.base_path}")
    print(f"  (expected structure: $agency/$name_$year.nc)")
    print(f"Output: {consolidator.output_base}")
    print(f"Strategy: trim_ends_interpolate_middle")
    print(f"Max NaN distance threshold: 2% of image width")
    print("="*80 + "\n")

    stats, lengths, agency_stats = consolidator.consolidate_all(dry_run=dry_run_mode)