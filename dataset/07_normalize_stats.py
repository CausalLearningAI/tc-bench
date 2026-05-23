"""
Compute normalization statistics for cyclone dataset with multiprocessing.

This script loads the entire training set in parallel and computes:
- Frame min/max for [0,1] normalization
- Pressure mean/std for z-score normalization
- Wind mean/std for z-score normalization

Uses Welford's parallel algorithm for numerical stability.
"""

import numpy as np
import json
from pathlib import Path
from datasets import load_from_disk
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple
import argparse
import os


def process_sample(args: Tuple[int, str]) -> Dict:
    """
    Process a single sample and return local statistics.
    
    Args:
        args: (index, dataset_path) tuple
        
    Returns:
        Dictionary with local statistics
    """
    idx, dataset_path = args
    
    # Load dataset (each worker loads it independently)
    dataset = load_from_disk(dataset_path)['train']
    sample = dataset[idx]
    
    # Extract data
    frames = np.array(sample['frames'], dtype=np.float32)
    pressure = np.array(sample['pressure'], dtype=np.float32)
    wind = np.array(sample['wind'], dtype=np.float32)
    
    # Filter NaN values
    valid_pressure = pressure[~np.isnan(pressure)]
    valid_wind = wind[~np.isnan(wind)]
    
    # Get cyclone_id if available
    cyclone_id = sample.get('cyclone_id', f'idx_{idx}')

    # Check for pixels < 180
    n_below_180 = int(np.sum(frames < 180))

    return {
        'idx': idx,
        'cyclone_id': cyclone_id,
        'frame_min': float(np.nanmin(frames)),
        'frame_max': float(np.nanmax(frames)),
        'frame_count': int(frames.size),
        'frame_below_180': n_below_180,

        'pressure_values': valid_pressure,
        'pressure_count': len(valid_pressure),
        'pressure_sum': float(np.sum(valid_pressure)),
        'pressure_sum_sq': float(np.sum(valid_pressure ** 2)),
        
        'wind_values': valid_wind,
        'wind_count': len(valid_wind),
        'wind_sum': float(np.sum(valid_wind)),
        'wind_sum_sq': float(np.sum(valid_wind ** 2)),
    }


def combine_statistics(results: list) -> Dict:
    """
    Combine statistics from multiple workers using parallel Welford's algorithm.
    
    Args:
        results: List of dictionaries with local statistics
        
    Returns:
        Dictionary with global statistics
    """
    print(f"  Combining statistics from {len(results)} samples...")
    
    # Initialize accumulators
    frame_min = float('inf')
    frame_max = float('-inf')
    frame_count = 0
    frame_below_180 = 0
    cyclones_with_below_180 = []  # Track cyclones with pixels < 180

    pressure_count = 0
    pressure_mean = 0.0
    pressure_M2 = 0.0
    
    wind_count = 0
    wind_mean = 0.0
    wind_M2 = 0.0
    
    # Process each result
    for i, res in enumerate(results):
        if i % 500 == 0 and i > 0:
            print(f"    Combined {i}/{len(results)} samples...")
        
        # Update frame stats
        frame_min = min(frame_min, res['frame_min'])
        frame_max = max(frame_max, res['frame_max'])
        frame_count += res['frame_count']
        frame_below_180 += res['frame_below_180']

        # Track cyclones with pixels < 180
        if res['frame_below_180'] > 0:
            cyclones_with_below_180.append({
                'idx': res['idx'],
                'cyclone_id': res['cyclone_id'],
                'n_pixels_below_180': res['frame_below_180'],
            })

        # Update pressure using Welford's parallel algorithm
        for p in res['pressure_values']:
            pressure_count += 1
            delta = p - pressure_mean
            pressure_mean += delta / pressure_count
            delta2 = p - pressure_mean
            pressure_M2 += delta * delta2
        
        # Update wind using Welford's parallel algorithm
        for w in res['wind_values']:
            wind_count += 1
            delta = w - wind_mean
            wind_mean += delta / wind_count
            delta2 = w - wind_mean
            wind_M2 += delta * delta2
    
    print(f"    Combined all {len(results)} samples.")
    
    # Compute final statistics
    pressure_std = float(np.sqrt(pressure_M2 / pressure_count)) if pressure_count > 1 else 0.0
    wind_std = float(np.sqrt(wind_M2 / wind_count)) if wind_count > 1 else 0.0
    
    return {
        'frame_min': frame_min,
        'frame_max': frame_max,
        'frame_count': frame_count,
        'frame_below_180': frame_below_180,
        'cyclones_with_below_180': cyclones_with_below_180,
        'pressure_mean': float(pressure_mean),
        'pressure_std': pressure_std,
        'pressure_count': pressure_count,
        'wind_mean': float(wind_mean),
        'wind_std': wind_std,
        'wind_count': wind_count,
    }


def main():
    parser = argparse.ArgumentParser(description='Compute normalization statistics with multiprocessing')
    data_root = os.environ.get("DATA_ROOT", os.path.join(os.environ["HOME"], "tcbench"))
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=os.path.join(data_root, 'dataset_hf'),
        help='Path to HuggingFace dataset'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of workers (default: all CPUs)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for normalization_stats.json (default: dataset_path/normalization_stats.json)'
    )
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    num_workers = args.num_workers or cpu_count()
    output_path = Path(args.output) if args.output else dataset_path / "normalization_stats.json"
    
    print(f"Computing normalization statistics for: {dataset_path}")
    print(f"Using {num_workers} worker processes")
    
    # Load dataset to get size
    print("\nLoading dataset metadata...")
    dataset = load_from_disk(str(dataset_path))
    train_size = len(dataset['train'])
    print(f"Training set size: {train_size} samples")
    
    # Create work items
    work_items = [(i, str(dataset_path)) for i in range(train_size)]
    
    # Process in parallel
    print(f"\nProcessing samples with {num_workers} workers...")
    with Pool(num_workers) as pool:
        results = pool.map(process_sample, work_items, chunksize=10)
    
    print(f"✓ Processed all {train_size} samples")
    
    # Combine statistics
    print("\nCombining statistics...")
    stats = combine_statistics(results)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPUTED STATISTICS")
    print("="*80)
    print(f"Frame range: [{stats['frame_min']:.2f}, {stats['frame_max']:.2f}] K")
    print(f"Total pixels: {stats['frame_count']:,}")
    print(f"Pixels < 180: {stats['frame_below_180']:,} ({100*stats['frame_below_180']/stats['frame_count']:.4f}%)")

    # Report cyclones with pixels < 180
    if stats['cyclones_with_below_180']:
        print(f"\nCyclones with pixels < 180 ({len(stats['cyclones_with_below_180'])} total):")
        # Sort by number of pixels below 180 (descending)
        sorted_cyclones = sorted(stats['cyclones_with_below_180'], key=lambda x: x['n_pixels_below_180'], reverse=True)
        for c in sorted_cyclones[:20]:  # Show top 20
            print(f"  idx={c['idx']}, cyclone_id={c['cyclone_id']}, n_pixels={c['n_pixels_below_180']:,}")
        if len(sorted_cyclones) > 20:
            print(f"  ... and {len(sorted_cyclones) - 20} more")

    print(f"\nPressure: mean={stats['pressure_mean']:.2f}, std={stats['pressure_std']:.2f}")
    print(f"  Total values: {stats['pressure_count']:,}")
    print(f"\nWind: mean={stats['wind_mean']:.2f}, std={stats['wind_std']:.2f}")
    print(f"  Total values: {stats['wind_count']:,}")
    
    # Save to file
    output_stats = {
        'frame_min': stats['frame_min'],
        'frame_max': stats['frame_max'],
        'pressure_mean': stats['pressure_mean'],
        'pressure_std': stats['pressure_std'],
        'wind_mean': stats['wind_mean'],
        'wind_std': stats['wind_std'],
    }
    
    print(f"\nSaving statistics to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(output_stats, f, indent=2)
    
    print("✓ Done!")


if __name__ == "__main__":
    main()
