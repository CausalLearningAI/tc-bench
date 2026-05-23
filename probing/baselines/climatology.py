#!/usr/bin/env python3
"""
Compute baseline statistics (per-basin, per-month means) using multiprocessing.
Usage: python compute_baselines_parallel.py --num_workers 24
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from datasets import load_from_disk
from multiprocessing import Pool, cpu_count
from functools import partial


def process_chunk(indices, dataset_path, split_name):
    """Process a chunk of cyclone indices and return statistics."""
    # Load dataset in worker process
    dataset = load_from_disk(dataset_path)
    split_dataset = dataset[split_name]

    # Track statistics for this chunk (including sum of squares for std)
    # Separate: per-basin and per-month (as listed in paper)
    basin_stats = defaultdict(lambda: {
        'pressure_sum': 0.0, 'pressure_sum_sq': 0.0, 'pressure_count': 0,
        'wind_sum': 0.0, 'wind_sum_sq': 0.0, 'wind_count': 0
    })
    month_stats = defaultdict(lambda: {
        'pressure_sum': 0.0, 'pressure_sum_sq': 0.0, 'pressure_count': 0,
        'wind_sum': 0.0, 'wind_sum_sq': 0.0, 'wind_count': 0
    })

    # Process each cyclone in this chunk
    for idx in indices:
        sample = split_dataset[idx]

        basin = sample['location']
        timestamps = sample['timestamps']
        pressure = np.array(sample['pressure'])
        wind = np.array(sample['wind'])

        # Extract month from first timestamp
        if len(timestamps) > 0:
            first_ts = str(timestamps[0])
            month = int(first_ts[5:7]) if len(first_ts) >= 7 else 0
        else:
            month = 0

        # Accumulate statistics (skip NaN values)
        for p, w in zip(pressure, wind):
            if not (np.isnan(p) or np.isnan(w)):
                # Per-basin stats
                basin_stats[basin]['pressure_sum'] += float(p)
                basin_stats[basin]['pressure_sum_sq'] += float(p * p)
                basin_stats[basin]['pressure_count'] += 1
                basin_stats[basin]['wind_sum'] += float(w)
                basin_stats[basin]['wind_sum_sq'] += float(w * w)
                basin_stats[basin]['wind_count'] += 1
                # Per-month stats (seasonal)
                month_stats[month]['pressure_sum'] += float(p)
                month_stats[month]['pressure_sum_sq'] += float(p * p)
                month_stats[month]['pressure_count'] += 1
                month_stats[month]['wind_sum'] += float(w)
                month_stats[month]['wind_sum_sq'] += float(w * w)
                month_stats[month]['wind_count'] += 1

    # Convert defaultdict to regular dict for serialization
    return {'basin': dict(basin_stats), 'month': dict(month_stats)}


def merge_stats(stats_list):
    """Merge statistics from multiple chunks."""
    merged_basin = defaultdict(lambda: {
        'pressure_sum': 0.0, 'pressure_sum_sq': 0.0, 'pressure_count': 0,
        'wind_sum': 0.0, 'wind_sum_sq': 0.0, 'wind_count': 0
    })
    merged_month = defaultdict(lambda: {
        'pressure_sum': 0.0, 'pressure_sum_sq': 0.0, 'pressure_count': 0,
        'wind_sum': 0.0, 'wind_sum_sq': 0.0, 'wind_count': 0
    })

    for stats in stats_list:
        # Merge basin stats
        for basin, values in stats['basin'].items():
            merged_basin[basin]['pressure_sum'] += values['pressure_sum']
            merged_basin[basin]['pressure_sum_sq'] += values['pressure_sum_sq']
            merged_basin[basin]['pressure_count'] += values['pressure_count']
            merged_basin[basin]['wind_sum'] += values['wind_sum']
            merged_basin[basin]['wind_sum_sq'] += values['wind_sum_sq']
            merged_basin[basin]['wind_count'] += values['wind_count']

        # Merge month stats
        for month, values in stats['month'].items():
            month_key = int(month) if isinstance(month, str) else month
            merged_month[month_key]['pressure_sum'] += values['pressure_sum']
            merged_month[month_key]['pressure_sum_sq'] += values['pressure_sum_sq']
            merged_month[month_key]['pressure_count'] += values['pressure_count']
            merged_month[month_key]['wind_sum'] += values['wind_sum']
            merged_month[month_key]['wind_sum_sq'] += values['wind_sum_sq']
            merged_month[month_key]['wind_count'] += values['wind_count']

    return {'basin': merged_basin, 'month': merged_month}


def compute_means(merged_stats):
    """Compute means and standard deviations from merged statistics."""
    basin_merged = merged_stats['basin']
    month_merged = merged_stats['month']

    # Per-basin means
    basin_means = {}
    for basin, s in basin_merged.items():
        if s['pressure_count'] > 0:
            p_mean = s['pressure_sum'] / s['pressure_count']
            p_std = np.sqrt(max(0, s['pressure_sum_sq'] / s['pressure_count'] - p_mean ** 2))
            w_mean = s['wind_sum'] / s['wind_count']
            w_std = np.sqrt(max(0, s['wind_sum_sq'] / s['wind_count'] - w_mean ** 2))

            basin_means[basin] = {
                'pressure': float(p_mean),
                'pressure_std': float(p_std),
                'wind': float(w_mean),
                'wind_std': float(w_std),
                'count': int(s['pressure_count'])
            }

    # Per-month means (seasonal)
    month_means = {}
    for month, s in month_merged.items():
        if s['pressure_count'] > 0:
            p_mean = s['pressure_sum'] / s['pressure_count']
            p_std = np.sqrt(max(0, s['pressure_sum_sq'] / s['pressure_count'] - p_mean ** 2))
            w_mean = s['wind_sum'] / s['wind_count']
            w_std = np.sqrt(max(0, s['wind_sum_sq'] / s['wind_count'] - w_mean ** 2))

            month_means[int(month)] = {
                'pressure': float(p_mean),
                'pressure_std': float(p_std),
                'wind': float(w_mean),
                'wind_std': float(w_std),
                'count': int(s['pressure_count'])
            }

    # Global means and stds (compute from basin stats - same total)
    global_pressure_sum = sum(s['pressure_sum'] for s in basin_merged.values())
    global_pressure_sum_sq = sum(s['pressure_sum_sq'] for s in basin_merged.values())
    global_wind_sum = sum(s['wind_sum'] for s in basin_merged.values())
    global_wind_sum_sq = sum(s['wind_sum_sq'] for s in basin_merged.values())
    global_count = sum(s['pressure_count'] for s in basin_merged.values())

    global_pressure_mean = global_pressure_sum / global_count if global_count > 0 else 0
    global_pressure_std = np.sqrt(max(0, global_pressure_sum_sq / global_count - global_pressure_mean ** 2)) if global_count > 0 else 0
    global_wind_mean = global_wind_sum / global_count if global_count > 0 else 0
    global_wind_std = np.sqrt(max(0, global_wind_sum_sq / global_count - global_wind_mean ** 2)) if global_count > 0 else 0

    return {
        'basin_means': basin_means,
        'month_means': month_means,
        'global_pressure': float(global_pressure_mean),
        'global_pressure_std': float(global_pressure_std),
        'global_wind': float(global_wind_mean),
        'global_wind_std': float(global_wind_std),
        'total_frames': int(global_count)
    }


def compute_statistics_parallel(dataset_path, split_name, num_workers):
    """Compute statistics using multiprocessing."""
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    split_dataset = dataset[split_name]
    n = len(split_dataset)
    
    print(f"Processing {n} cyclones with {num_workers} workers...")
    
    # Split indices into chunks for parallel processing
    chunk_size = max(10, n // (num_workers * 4))  # 4 chunks per worker for load balancing
    chunks = []
    for i in range(0, n, chunk_size):
        chunks.append(list(range(i, min(i + chunk_size, n))))
    
    print(f"Created {len(chunks)} chunks (chunk_size={chunk_size})")
    
    # Process chunks in parallel
    start_time = time.time()
    process_func = partial(process_chunk, dataset_path=dataset_path, split_name=split_name)
    
    with Pool(processes=num_workers) as pool:
        stats_list = pool.map(process_func, chunks)
    
    print(f"  Processing completed in {time.time() - start_time:.1f}s")
    
    # Merge results
    print("  Merging results...")
    merged_stats = merge_stats(stats_list)
    
    # Compute means
    print("  Computing means...")
    results = compute_means(merged_stats)
    
    print(f"✓ Completed: {results['total_frames']:,} frames, "
          f"{len(results['basin_means'])} basins, "
          f"{len(results['month_means'])} months")

    return results


def evaluate_baselines(train_stats, dataset_path):
    """Evaluate baseline predictions on actual test data frame-by-frame."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import pearsonr, spearmanr

    print("  Loading test dataset...")
    dataset = load_from_disk(dataset_path)
    test_dataset = dataset['test']

    # Collect actual predictions and ground truth frame-by-frame
    # Three baselines: global, basin (per-basin), month (per-month/seasonal)
    predictions = {
        'global': {'pressure': [], 'wind': []},
        'basin': {'pressure': [], 'wind': []},
        'month': {'pressure': [], 'wind': []}
    }
    ground_truth = {'pressure': [], 'wind': []}

    # Also track test set statistics
    test_pressure_values = []
    test_wind_values = []

    print(f"  Processing {len(test_dataset)} test cyclones...")

    # Process each cyclone in test set
    for sample in test_dataset:
        basin = sample['location']
        timestamps = sample['timestamps']
        pressure = np.array(sample['pressure'])
        wind = np.array(sample['wind'])

        # Extract month from first timestamp
        if len(timestamps) > 0:
            first_ts = str(timestamps[0])
            month = int(first_ts[5:7]) if len(first_ts) >= 7 else 0
        else:
            month = 0

        # Process each frame
        for p, w in zip(pressure, wind):
            if not (np.isnan(p) or np.isnan(w)):
                # Ground truth
                ground_truth['pressure'].append(float(p))
                ground_truth['wind'].append(float(w))
                test_pressure_values.append(float(p))
                test_wind_values.append(float(w))

                # Global mean prediction
                predictions['global']['pressure'].append(train_stats['global_pressure'])
                predictions['global']['wind'].append(train_stats['global_wind'])

                # Per-basin prediction
                basin_pred = train_stats['basin_means'].get(basin)
                if basin_pred:
                    predictions['basin']['pressure'].append(basin_pred['pressure'])
                    predictions['basin']['wind'].append(basin_pred['wind'])
                else:
                    # Fallback to global
                    predictions['basin']['pressure'].append(train_stats['global_pressure'])
                    predictions['basin']['wind'].append(train_stats['global_wind'])

                # Per-month prediction (seasonal)
                month_pred = train_stats['month_means'].get(month)
                if month_pred:
                    predictions['month']['pressure'].append(month_pred['pressure'])
                    predictions['month']['wind'].append(month_pred['wind'])
                else:
                    # Fallback to global
                    predictions['month']['pressure'].append(train_stats['global_pressure'])
                    predictions['month']['wind'].append(train_stats['global_wind'])

    # Convert to arrays and compute metrics
    gt_pressure = np.array(ground_truth['pressure'])
    gt_wind = np.array(ground_truth['wind'])

    # Compute test set statistics
    test_stats = {
        'pressure_mean': float(np.mean(test_pressure_values)),
        'pressure_std': float(np.std(test_pressure_values)),
        'wind_mean': float(np.mean(test_wind_values)),
        'wind_std': float(np.std(test_wind_values)),
        'n_frames': len(test_pressure_values)
    }

    results = {}
    for name in ['global', 'basin', 'month']:
        pred_p = np.array(predictions[name]['pressure'])
        pred_w = np.array(predictions[name]['wind'])

        # Compute all metrics
        results[name] = {
            'pressure': {
                'mae': float(mean_absolute_error(gt_pressure, pred_p)),
                'rmse': float(np.sqrt(mean_squared_error(gt_pressure, pred_p))),
                'r2': float(r2_score(gt_pressure, pred_p)),
                'pearson_r': float(pearsonr(gt_pressure, pred_p)[0]),
                'spearman_rho': float(spearmanr(gt_pressure, pred_p)[0])
            },
            'wind': {
                'mae': float(mean_absolute_error(gt_wind, pred_w)),
                'rmse': float(np.sqrt(mean_squared_error(gt_wind, pred_w))),
                'r2': float(r2_score(gt_wind, pred_w)),
                'pearson_r': float(pearsonr(gt_wind, pred_w)[0]),
                'spearman_rho': float(spearmanr(gt_wind, pred_w)[0])
            }
        }

    return results, test_stats


def main():
    parser = argparse.ArgumentParser(description='Compute baseline statistics in parallel')
    data_root = os.environ.get("DATA_ROOT", os.path.join(os.environ["HOME"], "tcbench"))
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.join(data_root, 'dataset_hf'),
                        help='Path to HuggingFace dataset')
    parser.add_argument('--num_workers', type=int, default=cpu_count(),
                        help='Number of parallel workers')
    parser.add_argument('--output_dir', type=str, default='logs',
                        help='Output directory for results')
    args = parser.parse_args()
    
    print(f"Starting baseline computation with {args.num_workers} workers")
    print("="*80)
    
    # Compute statistics for train and test splits
    print("\nComputing training statistics...")
    train_stats = compute_statistics_parallel(args.dataset_path, 'train', args.num_workers)
    
    # Evaluate baselines on actual test data
    print("\nEvaluating baselines on test set...")
    results, test_stats = evaluate_baselines(train_stats, args.dataset_path)
    
    print("\n" + "="*80)
    print("BASELINE RESULTS (Test Set)")
    print("="*80)
    
    # Print train statistics
    print(f"\nTRAIN STATISTICS:")
    print(f"  Pressure: {train_stats['global_pressure']:.2f} ± {train_stats['global_pressure_std']:.2f} hPa")
    print(f"  Wind: {train_stats['global_wind']:.2f} ± {train_stats['global_wind_std']:.2f} kt")
    print(f"  Total frames: {train_stats['total_frames']:,}")
    
    # Print test statistics
    print(f"\nTEST STATISTICS:")
    print(f"  Pressure: {test_stats['pressure_mean']:.2f} ± {test_stats['pressure_std']:.2f} hPa")
    print(f"  Wind: {test_stats['wind_mean']:.2f} ± {test_stats['wind_std']:.2f} kt")
    print(f"  Total frames: {test_stats['n_frames']:,}")
    
    name_map = {'global': 'Global Mean', 'basin': 'Basin Mean', 'month': 'Seasonal Mean'}
    for name in ['global', 'basin', 'month']:
        metrics = results[name]
        print(f"\n{name_map[name]}:")
        print(f"  PRESSURE - MAE: {metrics['pressure']['mae']:.3f}, RMSE: {metrics['pressure']['rmse']:.3f}, "
              f"R²: {metrics['pressure']['r2']:.3f}, Pearson: {metrics['pressure']['pearson_r']:.3f}, "
              f"Spearman: {metrics['pressure']['spearman_rho']:.3f}")
        print(f"  WIND     - MAE: {metrics['wind']['mae']:.3f}, RMSE: {metrics['wind']['rmse']:.3f}, "
              f"R²: {metrics['wind']['r2']:.3f}, Pearson: {metrics['wind']['pearson_r']:.3f}, "
              f"Spearman: {metrics['wind']['spearman_rho']:.3f}")

    # Print LaTeX table
    print("\n" + "="*80)
    print("LATEX TABLE FORMAT")
    print("="*80)
    print("\n% Pressure Prediction")
    for name in ['global', 'basin', 'month']:
        m = results[name]['pressure']
        print(f"{name_map[name]} & {m['mae']:.3f} & {m['rmse']:.3f} & {m['r2']:.3f} & {m['pearson_r']:.3f} & {m['spearman_rho']:.3f} \\\\")
    print("\n% Wind Prediction")
    for name in ['global', 'basin', 'month']:
        m = results[name]['wind']
        print(f"{name_map[name]} & {m['mae']:.3f} & {m['rmse']:.3f} & {m['r2']:.3f} & {m['pearson_r']:.3f} & {m['spearman_rho']:.3f} \\\\")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'baseline_statistics.json'
    output_data = {
        'train_stats': train_stats,
        'test_stats': test_stats,
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Also save normalization stats in the dataset directory format
    normalization_file = Path(args.dataset_path) / 'normalization_stats.json'
    normalization_stats = {
        'pressure_mean': train_stats['global_pressure'],
        'pressure_std': train_stats['global_pressure_std'],
        'wind_mean': train_stats['global_wind'],
        'wind_std': train_stats['global_wind_std'],
        'frame_min': 0.0,  # Placeholder - would need actual computation
        'frame_max': 1.0   # Placeholder - would need actual computation
    }
    
    with open(normalization_file, 'w') as f:
        json.dump(normalization_stats, f, indent=2)
    
    print(f"✓ Normalization stats saved to {normalization_file}")


if __name__ == '__main__':
    main()
