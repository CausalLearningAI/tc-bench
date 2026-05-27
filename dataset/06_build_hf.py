import xarray as xr
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Features, Value, Array2D, Array3D, Sequence
from typing import Dict, List, Optional
import argparse
from multiprocessing import Pool
import os


class CycloneDatasetBuilder:
    """
    Build a HuggingFace dataset from consolidated NetCDF cyclone files.
    """
    
    def __init__(self, consolidated_path: str, output_path: str, workers: int = 1):
        self.consolidated_path = Path(consolidated_path)
        self.output_path = Path(output_path)
        self.workers = workers
    
    def load_cyclone_from_netcdf(self, nc_file: Path) -> Optional[Dict]:
        """
        Load a single consolidated cyclone NetCDF file.
        
        Returns:
        --------
        Dict with:
            - cyclone_id: str
            - location: str
            - year: int
            - name: str
            - frames: np.ndarray (T, H, W)
            - pressure: np.ndarray (T,)
            - wind: np.ndarray (T,)
            - center: np.ndarray (T, 2) - [lat, lon]
            - timestamps: List[str]
            - num_timesteps: int
            - height: int
            - width: int
            - pressure_wind_valid: Optional[np.ndarray] (T,)
            - interpolated: bool
            - pressure_wind_strategy: str
        """
        try:
            ds = xr.open_dataset(nc_file)
            
            # Get basic metadata
            cyclone_name = ds.attrs['cyclone_name']
            year = ds.attrs['year']
            location = ds.attrs['location']
            
            # Load data arrays
            frames = ds["irwin_cdr"].values  # (T, H, W)
            frame_valid_mask = ds["frame_valid_mask"].values
            pressure = ds["pressure"].values  # (T,)
            wind = ds["wind"].values  # (T,)
            center = ds["center"].values  # (T, 2) - [lat, lon]
            timestamps = ds["time"].values.tolist()
            
            # Get validity mask if exists
            pressure_wind_valid = None
            if "pressure_wind_valid" in ds:
                pressure_wind_valid = ds["pressure_wind_valid"].values
            
            # # Parse metadata
            # metadata_list = []
            # for m in ds["metadata"].values:
            #     try:
            #         metadata_list.append(json.loads(m.item()))
            #     except:
            #         metadata_list.append({})
            
            example = {
                'cyclone_id': f"{location}_{year}_{cyclone_name}",
                'location': location,
                'year': int(year),
                'name': cyclone_name,
                'frames': frames.astype(np.float32),
                "frame_valid_mask": frame_valid_mask.astype(np.int8),
                'pressure': pressure.astype(np.float32),
                'wind': wind.astype(np.float32),
                'center': center.astype(np.float32),
                'timestamps': timestamps,
                'num_timesteps': frames.shape[0],
                'height': frames.shape[1],
                'width': frames.shape[2],
                'pressure_wind_valid': pressure_wind_valid.astype(np.int8) if pressure_wind_valid is not None else None,
                'interpolated': bool(ds.attrs.get('interpolated', 0)),
                'pressure_wind_strategy': ds.attrs.get('pressure_wind_strategy', 'unknown'),
            }
            
            ds.close()
            return example
            
        except Exception as e:
            print(f"Error loading {nc_file}: {e}")
            return None
    
    def scan_consolidated_files(self) -> List[Path]:
        """Scan for all consolidated NetCDF files."""
        nc_files = []
        
        for location_dir in self.consolidated_path.iterdir():
            if not location_dir.is_dir():
                continue
            
            for nc_file in location_dir.glob("*.nc"):
                nc_files.append(nc_file)
        
        return sorted(nc_files)
    
    @staticmethod
    def load_cyclone_from_netcdf_static(nc_file: Path) -> Optional[Dict]:
        """
        Static method to load a single consolidated cyclone NetCDF file.
        Used for multiprocessing to avoid pickling issues.
        
        Returns:
        --------
        Dict with cyclone data (same format as load_cyclone_from_netcdf)
        """
        try:
            ds = xr.open_dataset(nc_file)
            
            # Get basic metadata
            cyclone_name = ds.attrs['cyclone_name']
            year = ds.attrs['year']
            location = ds.attrs['location']
            
            # Load data arrays
            frames = ds["irwin_cdr"].values  # (T, H, W)
            frame_valid_mask = ds["frame_valid_mask"].values
            pressure = ds["pressure"].values  # (T,)
            wind = ds["wind"].values  # (T,)
            center = ds["center"].values  # (T, 2) - [lat, lon]
            timestamps = ds["time"].values.tolist()
            
            # Get validity mask if exists
            pressure_wind_valid = None
            if "pressure_wind_valid" in ds:
                pressure_wind_valid = ds["pressure_wind_valid"].values

            
            example = {
                'cyclone_id': f"{location}_{year}_{cyclone_name}",
                'location': location,
                'year': int(year),
                'name': cyclone_name,
                'frames': frames.astype(np.float32),
                "frame_valid_mask": frame_valid_mask.astype(np.int8),
                'pressure': pressure.astype(np.float32),
                'wind': wind.astype(np.float32),
                'center': center.astype(np.float32),
                'timestamps': timestamps,
                'num_timesteps': frames.shape[0],
                'height': frames.shape[1],
                'width': frames.shape[2],
                'pressure_wind_valid': pressure_wind_valid.astype(np.int8) if pressure_wind_valid is not None else None,
                'interpolated': bool(ds.attrs.get('interpolated', 0)),
                'pressure_wind_strategy': ds.attrs.get('pressure_wind_strategy', 'unknown'),
            }
            
            ds.close()
            return example
            
        except Exception as e:
            print(f"Error loading {nc_file}: {e}")
            return None
    
    def build_dataset(self, 
                     train_split: float = 0.8,
                     val_split: float = 0.1,
                     test_split: float = 0.1,
                     seed: int = 42) -> DatasetDict:
        """
        Build HuggingFace dataset from all consolidated files.
        
        Parameters:
        -----------
        train_split : float
            Fraction for training (default 0.8)
        val_split : float
            Fraction for validation (default 0.1)
        test_split : float
            Fraction for test (default 0.1)
        seed : int
            Random seed for splitting
        
        Returns:
        --------
        DatasetDict with train/validation/test splits
        """
        print("="*80)
        print("Building HuggingFace Dataset from Consolidated NetCDF Files")
        print("="*80)
        
        # Scan all files
        nc_files = self.scan_consolidated_files()
        print(f"\nFound {len(nc_files)} consolidated cyclone files")
        
        if len(nc_files) == 0:
            print("No files found!")
            return None
        
        # Collect statistics without loading everything into memory
        print("\nScanning cyclone metadata...")
        valid_files = []
        sequence_lengths = []
        locations = {}
        
        for nc_file in tqdm(nc_files, desc="Scanning files"):
            try:
                with xr.open_dataset(nc_file, engine='netcdf4') as ds:
                    num_timesteps = ds.dims.get('time', 0)
                    if num_timesteps > 0:
                        valid_files.append(nc_file)
                        sequence_lengths.append(num_timesteps)
                        
                        # Extract location from path
                        location = nc_file.parent.name
                        locations[location] = locations.get(location, 0) + 1
            except Exception as e:
                print(f"Error scanning {nc_file}: {e}")
                continue
        
        print(f"\nFound {len(valid_files)} valid cyclone files")
        
        if len(valid_files) == 0:
            print("No valid files found!")
            return None
        
        # Print statistics
        print(f"\nSequence length statistics:")
        print(f"  Min: {np.min(sequence_lengths)}")
        print(f"  Max: {np.max(sequence_lengths)}")
        print(f"  Mean: {np.mean(sequence_lengths):.1f}")
        print(f"  Median: {np.median(sequence_lengths):.1f}")
        
        print(f"\nCyclones by location:")
        for loc, count in sorted(locations.items()):
            print(f"  {loc}: {count}")
        
        # Create dataset with streaming generator to avoid loading all into memory
        print("\nCreating HuggingFace Dataset with streaming from disk...")
        if self.workers > 1:
            print(f"(Using {self.workers} workers for parallel loading)")
        else:
            print("(Loading cyclones on-demand to avoid memory overflow)")
        
        def example_generator():
            """Generator that loads cyclones on-demand from disk."""
            if self.workers > 1:
                # Use multiprocessing for parallel loading
                with Pool(self.workers) as pool:
                    # Use imap_unordered for memory-efficient streaming
                    for example in tqdm(
                        pool.imap_unordered(CycloneDatasetBuilder.load_cyclone_from_netcdf_static, valid_files),
                        total=len(valid_files),
                        desc="Streaming cyclones"
                    ):
                        if example is not None:
                            yield example
            else:
                # Sequential loading
                for nc_file in tqdm(valid_files, desc="Streaming cyclones"):
                    example = self.load_cyclone_from_netcdf(nc_file)
                    if example is not None:
                        yield example
        
        dataset = Dataset.from_generator(
            example_generator,
            writer_batch_size=200  # Write 200 cyclones at a time (safe with 400GB RAM)
        )
        
        # Split dataset
        print(f"\nSplitting dataset (train: {train_split}, val: {val_split}, test: {test_split})...")

        # Guard the degenerate case (e.g. single-storm smoke tests): if there
        # are too few trajectories to give every split at least one example,
        # keep them all in 'train'. Full builds (thousands of storms) are
        # unaffected and use the unchanged HuggingFace split below.
        n = len(dataset)
        n_val, n_test = round(n * val_split), round(n * test_split)
        if n < 3 or n_val < 1 or n_test < 1 or (n - n_val - n_test) < 1:
            print(f"[warn] only {n} trajectory(ies); requested split is "
                  "degenerate -> placing all in 'train'")
            dataset_dict = DatasetDict({'train': dataset})
        else:
            # First split: train vs (val+test)
            train_test = dataset.train_test_split(
                test_size=(val_split + test_split),
                seed=seed
            )

            # Second split: val vs test
            if val_split > 0 and test_split > 0:
                val_test = train_test['test'].train_test_split(
                    test_size=test_split / (val_split + test_split),
                    seed=seed
                )

                dataset_dict = DatasetDict({
                    'train': train_test['train'],
                    'validation': val_test['train'],
                    'test': val_test['test']
                })
            else:
                dataset_dict = DatasetDict({
                    'train': train_test['train'],
                    'test': train_test['test']
                })
        
        print("\n" + "="*80)
        print("Dataset Summary:")
        print("="*80)
        print(dataset_dict)
        
        # Show example
        print("\n" + "="*80)
        print("Example from training set:")
        print("="*80)
        example = dataset_dict['train'][0]
        print(f"Cyclone ID: {example['cyclone_id']}")
        print(f"Location: {example['location']}")
        print(f"Year: {example['year']}")
        print(f"Frames shape: {np.array(example['frames']).shape}")
        print(f"Pressure shape: {np.array(example['pressure']).shape}")
        print(f"Wind shape: {np.array(example['wind']).shape}")
        print(f"Timesteps: {example['num_timesteps']}")
        print(f"Spatial resolution: {example['height']}x{example['width']}")
        print(f"Pressure range: [{np.nanmin(example['pressure']):.1f}, {np.nanmax(example['pressure']):.1f}] mb")
        print(f"Wind range: [{np.nanmin(example['wind']):.1f}, {np.nanmax(example['wind']):.1f}] kts")
        
        if example['pressure_wind_valid'] is not None:
            n_valid = np.sum(example['pressure_wind_valid'])
            print(f"Valid pressure/wind: {n_valid}/{len(example['pressure_wind_valid'])}")
        
        return dataset_dict
    
    def save_dataset(self, dataset_dict: DatasetDict):
        """Save dataset to disk."""
        print(f"\n{'='*80}")
        print(f"Saving dataset to: {self.output_path}")
        print("="*80)
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(str(self.output_path))
        
        print("✓ Dataset saved successfully!")
        
        # Save dataset info
        info_path = self.output_path / "dataset_info.txt"
        with open(info_path, 'w') as f:
            f.write("Cyclone Dataset Information\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total samples: {len(dataset_dict['train']) + len(dataset_dict.get('validation', [])) + len(dataset_dict.get('test', []))}\n")
            f.write(f"Train: {len(dataset_dict['train'])}\n")
            if 'validation' in dataset_dict:
                f.write(f"Validation: {len(dataset_dict['validation'])}\n")
            if 'test' in dataset_dict:
                f.write(f"Test: {len(dataset_dict['test'])}\n")
            f.write(f"\nDataset structure:\n{dataset_dict}\n")
        
        print(f"✓ Dataset info saved to: {info_path}")
        
        return self.output_path


def load_and_test_dataset(dataset_path: str):
    """Load and test the saved dataset."""
    print("\n" + "="*80)
    print("Testing: Loading saved dataset")
    print("="*80)
    
    from datasets import load_from_disk
    
    dataset = load_from_disk(dataset_path)
    
    print(f"\n✓ Successfully loaded dataset from: {dataset_path}")
    print(dataset)
    
    # Test accessing data
    print("\nTesting data access...")
    sample = dataset['train'][0]
    print(f"  Cyclone ID: {sample['cyclone_id']}")
    print(f"  Frames shape: {np.array(sample['frames']).shape}")
    print(f"  Pressure values: {sample['pressure'][:5]} ...")
    
    # Test iteration
    print("\nTesting iteration...")
    count = 0
    for sample in dataset['train']:
        count += 1
        if count >= 3:
            break
    print(f"  ✓ Iterated through {count} samples successfully")
    
    return dataset


def push_to_hub(dataset_dict: DatasetDict, repo_id: str, private: bool = False, token: str = None):
    """
    Push dataset to HuggingFace Hub for public/private hosting.
    
    Parameters:
    -----------
    dataset_dict : DatasetDict
        The dataset to upload
    repo_id : str
        Repository ID on HuggingFace Hub (format: "username/dataset-name")
        Example: "your-username/cyclone-dataset"
    private : bool
        Whether to make the dataset private (default: False for public)
    token : str, optional
        HuggingFace API token. If not provided, will use cached token from `huggingface-cli login`
    
    Returns:
    --------
    str : URL to the dataset on HuggingFace Hub
    
    Usage:
    ------
    # First login (one time):
    # huggingface-cli login
    
    # Then push:
    push_to_hub(dataset_dict, "your-username/cyclone-dataset")
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return None
    
    print("\n" + "="*80)
    print(f"Pushing dataset to HuggingFace Hub: {repo_id}")
    print("="*80)
    
    # Push dataset
    print("\nUploading dataset... (this may take a while)")
    dataset_dict.push_to_hub(
        repo_id=repo_id,
        private=private,
        token=token
    )
    
    hub_url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"\n✓ Dataset successfully uploaded!")
    print(f"  View at: {hub_url}")
    print(f"\nTo load this dataset:")
    print(f"  from datasets import load_dataset")
    print(f"  dataset = load_dataset('{repo_id}')")
    
    return hub_url


def create_dataset_card(
    repo_id: str,
    dataset_dict: DatasetDict,
    description: str = None,
    license: str = "cc-by-4.0",
    citation: str = None,
    output_path: str = None
) -> str:
    """
    Create a README.md dataset card for HuggingFace Hub.
    
    Parameters:
    -----------
    repo_id : str
        Repository ID (e.g., "username/cyclone-dataset")
    dataset_dict : DatasetDict
        The dataset
    description : str, optional
        Custom description of the dataset
    license : str
        License identifier (default: "cc-by-4.0")
        Common options: "cc-by-4.0", "cc-by-sa-4.0", "mit", "apache-2.0"
    citation : str, optional
        BibTeX citation
    output_path : str, optional
        Where to save the README.md (if None, returns string)
    
    Returns:
    --------
    str : The dataset card content
    """
    # Collect dataset statistics
    n_train = len(dataset_dict['train'])
    n_val = len(dataset_dict.get('validation', []))
    n_test = len(dataset_dict.get('test', []))
    n_total = n_train + n_val + n_test
    
    example = dataset_dict['train'][0]
    
    # Get locations
    locations = set()
    for split in dataset_dict.values():
        for sample in split:
            locations.add(sample['location'])
    
    # Default description
    if description is None:
        description = """
This dataset contains satellite imagery and meteorological data for tropical cyclones 
from multiple ocean basins. Each cyclone includes a sequence of infrared brightness 
temperature images along with corresponding pressure and wind measurements.
"""
    
    # Default citation
    if citation is None:
        citation = f"""
@dataset{{{repo_id.replace('/', '_')},
  author = {{Your Name}},
  title = {{Cyclone Satellite Dataset}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{repo_id}}}}}
}}
"""
    
    card_content = f"""---
license: {license}
task_categories:
- time-series-forecasting
- image-classification
- video-classification
tags:
- climate
- weather
- tropical-cyclones
- satellite-imagery
- meteorology
size_categories:
- 1K<n<10K
---

# Cyclone Satellite Dataset

{description.strip()}

## Dataset Details

### Dataset Description

This dataset provides:
- **Satellite imagery**: Infrared brightness temperature (11 μm) at ~224×224 resolution
- **Meteorological data**: Central pressure (mb) and maximum wind speed (kts)
- **Temporal sequences**: 3-hourly observations spanning cyclone lifecycles
- **Multiple basins**: {', '.join(sorted(locations))}

### Dataset Sources

- **Data Provider**: NOAA FCDR (Fundamental Climate Data Record)
- **Repository**: [{repo_id}](https://huggingface.co/datasets/{repo_id})

## Dataset Structure

### Data Instances

Each instance represents one tropical cyclone with:

```python
{{
    'cyclone_id': 'tokyo_2023_YUN-YEUNG',
    'location': 'tokyo',
    'year': 2023,
    'name': 'YUN-YEUNG',
    'frames': List[List[List[float]]],  # Shape: (T, H, W)
    'pressure': List[float],  # Shape: (T,) in millibars
    'wind': List[float],  # Shape: (T,) in knots
    'timestamps': List[str],
    'num_timesteps': int,
    'height': 224,
    'width': 224,
    'pressure_wind_valid': List[int],  # 1=valid, 0=interpolated
}}
```

### Data Splits

- **Train**: {n_train} cyclones
- **Validation**: {n_val} cyclones
- **Test**: {n_test} cyclones
- **Total**: {n_total} cyclones

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `cyclone_id` | string | Unique identifier (format: location_year_name) |
| `frames` | array | Satellite infrared brightness temperature (Kelvin) |
| `pressure` | array | Central pressure (millibars) |
| `wind` | array | Maximum sustained wind speed (knots) |
| `timestamps` | array | ISO 8601 timestamps (3-hour intervals) |
| `pressure_wind_valid` | array | Validity mask (1=observed, 0=interpolated) |

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("{repo_id}")

# Access splits
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Convert sample to numpy
import numpy as np
sample = train_data[0]
frames = np.array(sample['frames'], dtype=np.float32)  # (T, H, W)
pressure = np.array(sample['pressure'], dtype=np.float32)  # (T,)
wind = np.array(sample['wind'], dtype=np.float32)  # (T,)
```

### PyTorch DataLoader

```python
import torch
from torch.utils.data import DataLoader

def collate_fn(batch):
    return {{
        'frames': [torch.tensor(x['frames']) for x in batch],
        'pressure': [torch.tensor(x['pressure']) for x in batch],
        'wind': [torch.tensor(x['wind']) for x in batch],
    }}

train_loader = DataLoader(
    dataset['train'],
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

for batch in train_loader:
    # batch['frames'] is a list of tensors with variable lengths
    pass
```

## Dataset Creation

### Source Data

- **Satellite**: NOAA FCDR Infrared brightness temperature (11 μm channel)
- **Best Track**: Agency best-track data (BOM, TOKYO, ATCF, etc.)
- **Preprocessing**: View zenith angle correction applied

### Data Processing

1. Individual NetCDF files consolidated per cyclone
2. NaN handling: leading/trailing missing values trimmed, in-between values interpolated
3. Native 3-hour temporal resolution preserved (no interpolation)
4. Spatial resolution: ~224×224 pixels centered on cyclone

## Considerations for Using the Data

### Temporal Characteristics

- **Variable length**: Sequences range from ~15-50 timesteps (45-150 hours)
- **Native cadence**: 3-hour intervals (no temporal interpolation)
- **Coverage**: Primarily mature phase (formation/dissipation trimmed)

### Spatial Characteristics

- **Resolution**: ~224×224 pixels
- **Domain**: Storm-centered, moving with cyclone
- **Variables**: Infrared brightness temperature (proxy for cloud-top temperature)

### Known Limitations

- Missing values possible in pressure/wind during weak phases
- Spatial resolution varies slightly by agency
- Satellite viewing angles may introduce artifacts
- Limited to storms with sufficient satellite coverage

## Citation

{citation.strip()}

## Dataset Card Contact

For questions or issues, please open an issue in the dataset repository.
"""
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(card_content)
        print(f"✓ Dataset card saved to: {output_path}")
    
    return card_content


def main():
    parser = argparse.ArgumentParser(
        description="Build HuggingFace dataset from consolidated cyclone NetCDF files"
    )
    data_root = os.environ.get("DATA_ROOT", os.path.join(os.environ["HOME"], "tcbench"))
    parser.add_argument(
        '--consolidated_path',
        type=str,
        default=os.path.join(data_root, 'consolidated'),
        help='Path to consolidated NetCDF files'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=os.path.join(data_root, 'dataset_hf'),
        help='Path to save HuggingFace dataset'
    )
    parser.add_argument(
        '--train_split',
        type=float,
        default=0.8,
        help='Training set fraction'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.1,
        help='Validation set fraction'
    )
    parser.add_argument(
        '--test_split',
        type=float,
        default=0.1,
        help='Test set fraction'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for splitting'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers for loading NetCDF files (default: 1)'
    )
    parser.add_argument(
        '--skip_test',
        action='store_true',
        help='Skip loading and testing the saved dataset'
    )
    parser.add_argument(
        '--push_to_hub',
        type=str,
        default=None,
        help='Push dataset to HuggingFace Hub (format: username/dataset-name)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make the HuggingFace dataset private (default: public)'
    )
    parser.add_argument(
        '--create_card',
        action='store_true',
        help='Create a dataset card (README.md)'
    )
    
    args = parser.parse_args()
    
    # Build dataset
    builder = CycloneDatasetBuilder(
        consolidated_path=args.consolidated_path,
        output_path=args.output_path,
        workers=args.workers
    )
    
    dataset_dict = builder.build_dataset(
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )
    
    if dataset_dict is None:
        print("Failed to build dataset!")
        return
    
    # Save dataset
    saved_path = builder.save_dataset(dataset_dict)
    
    # Create dataset card if requested
    if args.create_card or args.push_to_hub:
        print("\n" + "="*80)
        print("Creating dataset card...")
        print("="*80)
        card_path = saved_path / "README.md"
        repo_id = args.push_to_hub if args.push_to_hub else "username/cyclone-dataset"
        create_dataset_card(
            repo_id=repo_id,
            dataset_dict=dataset_dict,
            output_path=card_path
        )
    
    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        print("\n" + "="*80)
        print("Uploading to HuggingFace Hub...")
        print("="*80)
        print("\nNote: You must be logged in to HuggingFace Hub.")
        print("Run: huggingface-cli login")
        print()
        
        try:
            hub_url = push_to_hub(
                dataset_dict,
                repo_id=args.push_to_hub,
                private=args.private
            )
            
            if hub_url:
                print("\n" + "="*80)
                print("✓ Successfully uploaded to HuggingFace Hub!")
                print("="*80)
                print(f"  Dataset URL: {hub_url}")
                print(f"\nYou can now load this dataset from anywhere:")
                print(f"  from datasets import load_dataset")
                print(f"  dataset = load_dataset('{args.push_to_hub}')")
        except Exception as e:
            print(f"\n✗ Failed to upload to HuggingFace Hub: {e}")
            print("\nMake sure you are logged in:")
            print("  huggingface-cli login")
    
    # Test loading
    if not args.skip_test:
        loaded_dataset = load_and_test_dataset(str(saved_path))
    
    print("\n" + "="*80)
    print("✓ Smoke test complete!")
    print("="*80)
    print(f"\nDataset saved to: {saved_path}")
    print("\nNext steps:")
    print("1. Run consolidate_all() on full raw data")
    print("2. Build full HuggingFace dataset from consolidated files")
    print("3. Create PyTorch DataLoader with custom collation for variable-length sequences")


if __name__ == "__main__":
    main()
