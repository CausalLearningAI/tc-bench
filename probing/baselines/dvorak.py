#!/usr/bin/env python3
"""
Dvorak Technique Baseline for Cyclone Intensity Estimation

Implements a simplified version of the Dvorak Technique (Dvorak 1975, 1984),
a widely used operational method for estimating tropical cyclone intensity
from satellite infrared imagery.

The Dvorak Technique analyzes cloud patterns in satellite images to determine:
1. Storm center location
2. Cloud pattern type (eye, central dense overcast, shear, etc.)
3. Intensity estimate based on pattern characteristics

Reference:
- Dvorak, V. F. (1975). "Tropical cyclone intensity analysis and forecasting from satellite imagery."
- Dvorak, V. F. (1984). "Tropical Cyclone Intensity Analysis Using Satellite Data."
"""

import numpy as np
import argparse
import json
import os
import time
from pathlib import Path
from datasets import load_from_disk
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import ndimage
from collections import defaultdict
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


class DvorakTechnique:
    """
    Simplified implementation of the Dvorak Technique for cyclone intensity estimation.
    
    The technique analyzes infrared satellite imagery to identify cloud patterns
    and estimate intensity based on:
    - Cloud organization and symmetry
    - Central dense overcast (CDO) characteristics
    - Eye features (if present)
    - Banding patterns
    - Temperature characteristics
    """
    
    def __init__(self, 
                 min_temp=180.0,  # Minimum brightness temperature (K)
                 max_temp=320.0,  # Maximum brightness temperature (K)
                 eye_temp_threshold=270.0):  # Temperature threshold for eye detection
        """
        Initialize Dvorak Technique parameters.
        
        Args:
            min_temp: Minimum expected brightness temperature
            max_temp: Maximum expected brightness temperature
            eye_temp_threshold: Temperature above which we consider it a potential eye
        """
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.eye_temp_threshold = eye_temp_threshold
        
        # Dvorak T-number to intensity mappings (approximate)
        # T-number ranges from 1.0 to 8.0
        self.t_number_to_pressure = self._build_t_number_pressure_table()
        self.t_number_to_wind = self._build_t_number_wind_table()
    
    def _build_t_number_pressure_table(self):
        """Build T-number to central pressure lookup table (hPa)."""
        # Approximate Dvorak T-number to pressure relationship
        return {
            1.0: 1009, 1.5: 1005, 2.0: 1000, 2.5: 997, 3.0: 991,
            3.5: 984, 4.0: 976, 4.5: 966, 5.0: 954, 5.5: 941,
            6.0: 927, 6.5: 914, 7.0: 898, 7.5: 879, 8.0: 858
        }
    
    def _build_t_number_wind_table(self):
        """Build T-number to maximum wind speed lookup table (knots)."""
        # Approximate Dvorak T-number to wind relationship
        return {
            1.0: 25, 1.5: 25, 2.0: 30, 2.5: 35, 3.0: 45,
            3.5: 55, 4.0: 65, 4.5: 77, 5.0: 90, 5.5: 102,
            6.0: 115, 6.5: 127, 7.0: 140, 7.5: 155, 8.0: 170
        }
    
    def analyze_image(self, image):
        """
        Analyze a single satellite image and estimate intensity.
        
        Args:
            image: 2D numpy array of brightness temperatures (in Kelvin)
            
        Returns:
            dict with 'pressure' and 'wind' estimates
        """
        if image is None or image.size == 0:
            return {'pressure': 1000.0, 'wind': 30.0, 't_number': 2.0}
        
        # Handle NaN values
        image_clean = np.copy(image)
        if np.any(np.isnan(image_clean)):
            image_clean = np.nan_to_num(image_clean, nan=np.nanmean(image_clean))
        
        # 1. Cloud Top Temperature Analysis
        min_cloud_temp = np.min(image_clean)
        mean_cloud_temp = np.mean(image_clean)
        
        # 2. Cloud Pattern Organization
        # Measure symmetry and organization using standard deviation
        center_region = self._extract_center_region(image_clean)
        pattern_symmetry = self._measure_symmetry(center_region)
        
        # 3. Eye Detection
        eye_present, eye_score = self._detect_eye(image_clean)
        
        # 4. Central Dense Overcast (CDO) Analysis
        cdo_strength = self._analyze_cdo(image_clean)
        
        # 5. Compute T-number based on features
        t_number = self._compute_t_number(
            min_cloud_temp=min_cloud_temp,
            mean_cloud_temp=mean_cloud_temp,
            pattern_symmetry=pattern_symmetry,
            eye_present=eye_present,
            eye_score=eye_score,
            cdo_strength=cdo_strength
        )
        
        # 6. Convert T-number to pressure and wind
        pressure = self._t_number_to_pressure(t_number)
        wind = self._t_number_to_wind(t_number)
        
        return {
            'pressure': pressure,
            'wind': wind,
            't_number': t_number,
            'min_temp': min_cloud_temp,
            'mean_temp': mean_cloud_temp,
            'temp_enhancement': mean_cloud_temp - min_cloud_temp,
            'eye_present': eye_present
        }
    
    def _extract_center_region(self, image, fraction=0.4):
        """Extract the center region of the image for analysis."""
        h, w = image.shape
        center_h, center_w = h // 2, w // 2
        radius_h = int(h * fraction / 2)
        radius_w = int(w * fraction / 2)
        
        return image[
            max(0, center_h - radius_h):min(h, center_h + radius_h),
            max(0, center_w - radius_w):min(w, center_w + radius_w)
        ]
    
    def _measure_symmetry(self, region):
        """
        Measure the symmetry/organization of cloud patterns.
        Higher values indicate more organized systems.
        """
        if region.size == 0:
            return 0.0
        
        # Use coefficient of variation as a measure of organization
        # More organized systems have lower variance relative to mean
        std = np.std(region)
        mean = np.mean(region)
        
        if mean > 0:
            cv = std / mean
            # Convert to symmetry score (0 to 1, higher is more symmetric)
            symmetry = 1.0 / (1.0 + cv)
        else:
            symmetry = 0.0
        
        return symmetry
    
    def _detect_eye(self, image):
        """
        Detect the presence of an eye in the cyclone.
        
        Returns:
            tuple: (eye_present: bool, eye_score: float)
        """
        center_region = self._extract_center_region(image, fraction=0.2)
        
        if center_region.size == 0:
            return False, 0.0
        
        # Eye is characterized by warmer temperatures in the center
        center_mean = np.mean(center_region)
        
        # Compare to surrounding annulus
        h, w = image.shape
        center_h, center_w = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        
        # Inner circle (eye) and outer annulus
        inner_radius = min(h, w) * 0.1
        outer_radius = min(h, w) * 0.25
        
        distance = np.sqrt((y - center_h)**2 + (x - center_w)**2)
        annulus_mask = (distance >= inner_radius) & (distance < outer_radius)
        
        if np.any(annulus_mask):
            annulus_mean = np.mean(image[annulus_mask])
            
            # Eye should be warmer than surroundings
            temp_diff = center_mean - annulus_mean
            
            # Eye detection criteria
            eye_present = (temp_diff > 10.0) and (center_mean > self.eye_temp_threshold)
            eye_score = max(0.0, min(1.0, temp_diff / 50.0))
            
            return eye_present, eye_score
        
        return False, 0.0
    
    def _analyze_cdo(self, image):
        """
        Analyze Central Dense Overcast characteristics.
        
        Returns:
            float: CDO strength score (0 to 1)
        """
        center_region = self._extract_center_region(image, fraction=0.5)
        
        if center_region.size == 0:
            return 0.0
        
        # CDO is characterized by extensive cold cloud tops
        cold_threshold = 220.0  # Kelvin
        cold_fraction = np.sum(center_region < cold_threshold) / center_region.size
        
        # CDO strength based on extent of cold clouds
        cdo_strength = min(1.0, cold_fraction * 2.0)
        
        return cdo_strength
    
    def _compute_t_number(self, min_cloud_temp, mean_cloud_temp,
                         pattern_symmetry, eye_present, eye_score, cdo_strength):
        """
        Compute Dvorak T-number based on multiple features.

        T-number ranges from 1.0 (weak) to 8.0 (extremely intense).

        Key insight: Real Dvorak uses temperature ENHANCEMENT (contrast between
        coldest clouds and warmer surroundings), not absolute minimum temperature.
        Cold cloud tops are common in ALL tropical cyclones; what matters is the
        temperature contrast indicating deep convection strength.
        """
        # Compute temperature enhancement (contrast)
        # This measures how much colder the coldest clouds are vs the mean
        temp_enhancement = mean_cloud_temp - min_cloud_temp

        # Base T-number from temperature enhancement
        # Typical enhancements range from 20K (weak) to 100K+ (intense)
        # Calibrated so average cyclone (~60K enhancement) gets T ~2.5-3
        if temp_enhancement > 100:
            base_t = 6.0
        elif temp_enhancement > 90:
            base_t = 5.0
        elif temp_enhancement > 80:
            base_t = 4.0
        elif temp_enhancement > 70:
            base_t = 3.5
        elif temp_enhancement > 60:
            base_t = 3.0
        elif temp_enhancement > 50:
            base_t = 2.5
        elif temp_enhancement > 40:
            base_t = 2.0
        elif temp_enhancement > 30:
            base_t = 1.5
        else:
            base_t = 1.0

        # Adjust based on organization/symmetry (smaller effect)
        # Symmetry typically ranges 0.7-0.95, center around 0.85
        symmetry_adjustment = (pattern_symmetry - 0.85) * 2.0

        # Adjust for eye presence (major intensity indicator)
        # Eye presence strongly indicates intense storm
        if eye_present:
            eye_adjustment = 1.5 + eye_score * 0.5
        else:
            eye_adjustment = 0.0

        # Adjust for CDO extent (smaller effect)
        # CDO strength ranges 0-1, typical is 0.3-0.7
        cdo_adjustment = (cdo_strength - 0.5) * 0.5

        # Combine adjustments
        t_number = base_t + symmetry_adjustment + eye_adjustment + cdo_adjustment

        # Constrain to valid range
        t_number = max(1.0, min(8.0, t_number))

        return t_number
    
    def _t_number_to_pressure(self, t_number):
        """Convert T-number to central pressure using linear interpolation."""
        t_values = sorted(self.t_number_to_pressure.keys())
        
        # Find bracketing T-numbers
        if t_number <= t_values[0]:
            return self.t_number_to_pressure[t_values[0]]
        if t_number >= t_values[-1]:
            return self.t_number_to_pressure[t_values[-1]]
        
        # Linear interpolation
        for i in range(len(t_values) - 1):
            if t_values[i] <= t_number <= t_values[i+1]:
                t_low, t_high = t_values[i], t_values[i+1]
                p_low = self.t_number_to_pressure[t_low]
                p_high = self.t_number_to_pressure[t_high]
                
                # Interpolate
                frac = (t_number - t_low) / (t_high - t_low)
                pressure = p_low + frac * (p_high - p_low)
                return pressure
        
        return 1000.0  # Default
    
    def _t_number_to_wind(self, t_number):
        """Convert T-number to maximum wind speed using linear interpolation."""
        t_values = sorted(self.t_number_to_wind.keys())
        
        # Find bracketing T-numbers
        if t_number <= t_values[0]:
            return self.t_number_to_wind[t_values[0]]
        if t_number >= t_values[-1]:
            return self.t_number_to_wind[t_values[-1]]
        
        # Linear interpolation
        for i in range(len(t_values) - 1):
            if t_values[i] <= t_number <= t_values[i+1]:
                t_low, t_high = t_values[i], t_values[i+1]
                w_low = self.t_number_to_wind[t_low]
                w_high = self.t_number_to_wind[t_high]
                
                # Interpolate
                frac = (t_number - t_low) / (t_high - t_low)
                wind = w_low + frac * (w_high - w_low)
                return wind
        
        return 30.0  # Default


def process_single_cyclone(idx, split_dataset):
    """
    Process a single cyclone with the Dvorak Technique.
    
    Args:
        idx: Index in the dataset
        split_dataset: HuggingFace dataset split
        
    Returns:
        tuple: (predictions, ground_truth, metadata)
    """
    dvorak = DvorakTechnique()
    
    sample = split_dataset[idx]
    frames = sample['frames']
    pressure_gt = sample['pressure']
    wind_gt = sample['wind']
    
    predictions = {'pressure': [], 'wind': []}
    ground_truth = {'pressure': [], 'wind': []}
    metadata = {
        't_numbers': [],
        'min_temps': [],
        'temp_enhancements': [],
        'eye_detected': []
    }
    
    # Process each frame
    for k, frame in enumerate(frames):
        if frame is not None and len(frame) > 0:
            # Analyze with Dvorak Technique
            result = dvorak.analyze_image(np.array(frame))
            
            predictions['pressure'].append(result['pressure'])
            predictions['wind'].append(result['wind'])
            
            # Record metadata
            metadata['t_numbers'].append(result['t_number'])
            metadata['min_temps'].append(result['min_temp'])
            metadata['temp_enhancements'].append(result['temp_enhancement'])
            metadata['eye_detected'].append(result['eye_present'])
        else:
            # Use default values for missing frames
            predictions['pressure'].append(1000.0)
            predictions['wind'].append(30.0)
            metadata['t_numbers'].append(2.0)
            metadata['min_temps'].append(280.0)
            metadata['temp_enhancements'].append(0.0)
            metadata['eye_detected'].append(False)
        
        # Ground truth
        if k < len(pressure_gt) and k < len(wind_gt):
            ground_truth['pressure'].append(pressure_gt[k])
            ground_truth['wind'].append(wind_gt[k])
    
    return predictions, ground_truth, metadata


def process_dataset_split(split_dataset, num_workers=None, desc="Processing"):
    """
    Process a dataset split with the Dvorak Technique using multiprocessing.
    
    Args:
        split_dataset: HuggingFace dataset split
        num_workers: Number of parallel workers (default: cpu_count)
        desc: Description for progress bar
        
    Returns:
        dict: Predictions and ground truth
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 24)  # Cap at 24 workers
    
    n = len(split_dataset)
    indices = list(range(n))
    
    print(f"Processing {n} cyclones with {num_workers} workers...")
    
    # Process in parallel
    process_func = partial(process_single_cyclone, split_dataset=split_dataset)
    
    all_predictions = {'pressure': [], 'wind': []}
    all_ground_truth = {'pressure': [], 'wind': []}
    all_metadata = {
        't_numbers': [],
        'min_temps': [],
        'temp_enhancements': [],
        'eye_detected': []
    }
    
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, indices),
            total=n,
            desc=desc
        ))
    
    # Aggregate results
    for predictions, ground_truth, metadata in results:
        all_predictions['pressure'].extend(predictions['pressure'])
        all_predictions['wind'].extend(predictions['wind'])
        all_ground_truth['pressure'].extend(ground_truth['pressure'])
        all_ground_truth['wind'].extend(ground_truth['wind'])
        all_metadata['t_numbers'].extend(metadata['t_numbers'])
        all_metadata['min_temps'].extend(metadata['min_temps'])
        all_metadata['temp_enhancements'].extend(metadata['temp_enhancements'])
        all_metadata['eye_detected'].extend(metadata['eye_detected'])
    
    return all_predictions, all_ground_truth, all_metadata


def evaluate_predictions(predictions, ground_truth):
    """
    Evaluate Dvorak predictions against ground truth.
    
    Args:
        predictions: Dict with 'pressure' and 'wind' predictions
        ground_truth: Dict with 'pressure' and 'wind' ground truth
        
    Returns:
        dict: Evaluation metrics
    """
    # Filter out NaN values
    pred_p = np.array(predictions['pressure'])
    pred_w = np.array(predictions['wind'])
    gt_p = np.array(ground_truth['pressure'])
    gt_w = np.array(ground_truth['wind'])
    
    # Create mask for valid values
    valid_mask = ~(np.isnan(pred_p) | np.isnan(pred_w) | 
                   np.isnan(gt_p) | np.isnan(gt_w))
    
    if np.sum(valid_mask) == 0:
        return {
            'pressure_rmse': np.nan,
            'pressure_mae': np.nan,
            'wind_rmse': np.nan,
            'wind_mae': np.nan,
            'n_samples': 0
        }
    
    pred_p = pred_p[valid_mask]
    pred_w = pred_w[valid_mask]
    gt_p = gt_p[valid_mask]
    gt_w = gt_w[valid_mask]
    
    metrics = {
        'pressure_rmse': np.sqrt(mean_squared_error(gt_p, pred_p)),
        'pressure_mae': mean_absolute_error(gt_p, pred_p),
        'wind_rmse': np.sqrt(mean_squared_error(gt_w, pred_w)),
        'wind_mae': mean_absolute_error(gt_w, pred_w),
        'n_samples': len(pred_p)
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Dvorak Technique Baseline')
    data_root = os.environ.get("DATA_ROOT", os.path.join(os.environ["HOME"], "tcbench"))
    parser.add_argument('--dataset_path', type=str,
                       default=os.path.join(data_root, 'dataset_hf'),
                       help='Path to HuggingFace dataset')
    parser.add_argument('--output_path', type=str,
                       default='logs/dvorak_results.json',
                       help='Path to save results')
    parser.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'],
                       help='Dataset splits to evaluate')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of parallel workers (default: auto)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("DVORAK TECHNIQUE BASELINE")
    print("="*80)
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_path}")
    print(f"Workers: {args.num_workers or 'auto'}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk(args.dataset_path)
    print(f"✓ Dataset loaded")
    print(f"  Train: {len(dataset['train'])} samples")
    if 'validation' in dataset:
        print(f"  Validation: {len(dataset['validation'])} samples")
    if 'test' in dataset:
        print(f"  Test: {len(dataset['test'])} samples")
    print()
    
    # Process each split
    results = {}
    
    for split_name in args.splits:
        if split_name not in dataset:
            print(f"Warning: Split '{split_name}' not found in dataset. Skipping.")
            continue
        
        print(f"\nProcessing {split_name} set...")
        start_time = time.time()
        
        predictions, ground_truth, metadata = process_dataset_split(
            dataset[split_name],
            num_workers=args.num_workers,
            desc=f"Analyzing {split_name}"
        )
        
        # Evaluate
        metrics = evaluate_predictions(predictions, ground_truth)
        
        elapsed = time.time() - start_time
        
        print(f"\n{split_name.upper()} RESULTS:")
        print(f"  Pressure RMSE: {metrics['pressure_rmse']:.2f} hPa")
        print(f"  Pressure MAE:  {metrics['pressure_mae']:.2f} hPa")
        print(f"  Wind RMSE:     {metrics['wind_rmse']:.2f} kt")
        print(f"  Wind MAE:      {metrics['wind_mae']:.2f} kt")
        print(f"  Samples:       {metrics['n_samples']:,}")
        print(f"  Time:          {elapsed:.1f}s")
        
        # Additional statistics
        eye_detection_rate = np.mean(metadata['eye_detected'])
        mean_t_number = np.mean(metadata['t_numbers'])
        mean_min_temp = np.mean(metadata['min_temps'])
        mean_temp_enhancement = np.mean(metadata['temp_enhancements'])

        print(f"\nDvorak Analysis Statistics:")
        print(f"  Eye Detection Rate:     {eye_detection_rate*100:.1f}%")
        print(f"  Mean T-number:          {mean_t_number:.2f}")
        print(f"  Mean Min Temp:          {mean_min_temp:.1f}K")
        print(f"  Mean Temp Enhancement:  {mean_temp_enhancement:.1f}K")

        results[split_name] = {
            'metrics': metrics,
            'statistics': {
                'eye_detection_rate': float(eye_detection_rate),
                'mean_t_number': float(mean_t_number),
                'mean_min_temp': float(mean_min_temp),
                'mean_temp_enhancement': float(mean_temp_enhancement)
            },
            'time': elapsed
        }
    
    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print(f"✓ Results saved to {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
