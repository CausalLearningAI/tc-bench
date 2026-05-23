#!/usr/bin/env python3
"""
Create OOD (Out-of-Distribution) basin splits from an existing Hugging Face feature dataset.
For each basin, creates a new dataset with that basin as the OOD test set, and all other basins as train+val.

Usage:
    python scripts/create_ood_basin_split.py --feature_dataset /path/to/features_mae \
        --output_dir /path/to/output_ood_splits

Options:
    --basin_field location   # or 'basin' if your dataset uses that field
    --val_frac 0.1           # fraction of train+val to use as validation
"""
import argparse
from pathlib import Path
from datasets import load_from_disk, DatasetDict, Dataset
import numpy as np

parser = argparse.ArgumentParser(description="Create OOD basin splits from HF feature dataset.")
parser.add_argument('--feature_dataset', type=str, required=True, help='Path to input HF feature dataset')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save OOD splits')
parser.add_argument('--basin_field', type=str, default='location', help='Field name for basin/region (default: location)')
parser.add_argument('--val_frac', type=float, default=0.1, help='Fraction of train+val for validation split')
args = parser.parse_args()

# Load dataset
print(f"Loading dataset from {args.feature_dataset}")
ds = load_from_disk(args.feature_dataset)

# Combine all splits for basin filtering
all_examples = []
for split in ds.keys():
    for ex in ds[split]:
        all_examples.append(ex)

# Get all unique basins
basins = sorted(set(ex[args.basin_field] for ex in all_examples))
print(f"Found basins: {basins}")

# Found basins: ['atcf', 'bom', 'hurdat_atl', 'hurdat_epa', 'nadi', 'newdelhi', 'reunion', 'tokyo', 'wellington']

for basin in basins: #basins: 
    # skip basin if already exists
    out_path = Path(args.output_dir) / f"ood_basin_{basin}"
    if out_path.exists():
        print(f"Skipping basin '{basin}' because output path {out_path} already exists.")
        continue

    print(f"\n=== Creating OOD split: {basin} as test ===")
    # OOD test set: all examples from this basin
    test_examples = [ex for ex in all_examples if ex[args.basin_field] == basin]
    # Train+val: all other basins
    trainval_examples = [ex for ex in all_examples if ex[args.basin_field] != basin]

    # Shuffle trainval for reproducibility
    rng = np.random.default_rng(42)
    rng.shuffle(trainval_examples)

    # Split train/val
    n_val = int(len(trainval_examples) * args.val_frac)
    val_examples = trainval_examples[:n_val]
    train_examples = trainval_examples[n_val:]

    # Warn and skip if any split is empty
    if len(train_examples) == 0 or len(val_examples) == 0 or len(test_examples) == 0:
        print(f"[WARNING] Skipping basin '{basin}' because one or more splits are empty: "
              f"train={len(train_examples)}, val={len(val_examples)}, test={len(test_examples)}")
        continue

    # Build new DatasetDict
    dsdict = DatasetDict({
        'train': Dataset.from_list(train_examples),
        'validation': Dataset.from_list(val_examples),
        'test': Dataset.from_list(test_examples),
    })

    # Save to disk
    out_path = Path(args.output_dir) / f"ood_basin_{basin}"
    print(f"Saving OOD split to {out_path}")
    dsdict.save_to_disk(str(out_path))

print("\nAll OOD splits created!")
