"""
Base class for cyclone intensity regression models.

Provides common training logic, metrics, and utilities for benchmarking
different architectures on cyclone intensity prediction.
"""

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


class BaseIntensityRegressor(LightningModule):
    """
    Abstract base class for cyclone intensity regression models.
    
    Subclasses must implement:
    - forward(): Define model forward pass
    - configure_optimizers(): Define optimizer and scheduler
    """
    
    def __init__(
        self,
        input_variable: str = "features",  # "frames", "features", etc.
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        target_variable: str = "pressure",  # "pressure", , or "both"
        loss_fn: str = "mse",  # "mse", "mae", "huber"
        compile_model: bool = False,  # torch.compile for speedup
        output_csv_path: Optional[str] = None,  # Path to save test predictions CSV
        model_name: str = "model",  # Model name for CSV output
        pressure_mean: Optional[float] = None,  # Normalization stats for denormalization
        pressure_std: Optional[float] = None,
        wind_mean: Optional[float] = None,
        wind_std: Optional[float] = None,
        **kwargs
    ):
        """
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            target_variable: Which variable to predict
            loss_fn: Loss function to use
            compile_model: Whether to use torch.compile() for faster training
            output_csv_path: If provided, save test predictions to this CSV path
            model_name: Model name to include in CSV output
            pressure_mean: Mean pressure for denormalization
            pressure_std: Std pressure for denormalization
            wind_mean: Mean wind for denormalization
            wind_std: Std wind for denormalization
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_variable = input_variable
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.target_variable = target_variable
        self.compile_model = compile_model
        self.output_csv_path = output_csv_path
        self.model_name = model_name

        # Store normalization stats for denormalization
        if pressure_mean is not None:
            self.pressure_mean = pressure_mean
            self.pressure_std = pressure_std
        if wind_mean is not None:
            self.wind_mean = wind_mean
            self.wind_std = wind_std
        
        # Loss function
        if loss_fn == "mse":
            self.criterion = nn.MSELoss()
        elif loss_fn == "mae":
            self.criterion = nn.L1Loss()
        elif loss_fn == "huber":
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
        
        # Metrics for each target variable
        self.train_pressure_rmse = MeanSquaredError(squared=False)
        
        self.val_pressure_rmse = MeanSquaredError(squared=False)
        
        
        self.test_pressure_rmse = MeanSquaredError(squared=False)
        
        # Store predictions for plotting
        self.validation_outputs = []
        self.test_outputs = []
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass - must be implemented by subclasses.
        
        Args:
            x: Input frames, shape (B, C, H, W) for single-frame
               or (B, T, C, H, W) for sequences
        
        Returns:
            Dict with keys:
                - "pressure": (B,) or (B, T) tensor of pressure predictions
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def _compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute loss based on target variable(s)."""
        if self.target_variable == "pressure":
            return self.criterion(predictions["pressure"], targets["pressure"])
        
        elif self.target_variable == "wind":
            return self.criterion(predictions["wind"], targets["wind"])
        
        elif self.target_variable == "both":
            loss_pressure = self.criterion(predictions["pressure"], targets["pressure"])
            loss_wind = self.criterion(predictions["wind"], targets["wind"])
            return loss_pressure + loss_wind
        
        else:
            raise ValueError(f"Unknown target variable: {self.target_variable}")
    
    def _denormalize_targets(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        """Denormalize predictions and targets for metric computation."""
        denorm_pred = {}
        denorm_targets = {}
        
        if "pressure" in predictions:
            if hasattr(self, 'pressure_mean') and hasattr(self, 'pressure_std'):
                denorm_pred["pressure"] = predictions["pressure"] * self.pressure_std + self.pressure_mean
                denorm_targets["pressure"] = targets["pressure"] * self.pressure_std + self.pressure_mean
            else:
                denorm_pred["pressure"] = predictions["pressure"]
                denorm_targets["pressure"] = targets["pressure"]
        
        if "wind" in predictions:
            if hasattr(self, 'wind_mean') and hasattr(self, 'wind_std'):
                denorm_pred["wind"] = predictions["wind"] * self.wind_std + self.wind_mean
                denorm_targets["wind"] = targets["wind"] * self.wind_std + self.wind_mean
            else:
                denorm_pred["wind"] = predictions["wind"]
                denorm_targets["wind"] = targets["wind"]
        
        return denorm_pred, denorm_targets
    
    def _update_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        stage: str  # "train", "val", or "test"
    ):
        """Update metrics for current stage (in denormalized space)."""
        
        # Denormalize for metrics (keep loss in normalized space)
        denorm_pred, denorm_targets = self._denormalize_targets(predictions, targets)
        
        # Get metric objects for current stage
        if stage == "train":
            pressure_rmse = self.train_pressure_rmse
        elif stage == "val":
            pressure_rmse = self.val_pressure_rmse
        else:  # test
            pressure_rmse = self.test_pressure_rmse
                
        # Update metrics with denormalized values
        if "pressure" in denorm_pred:
            pressure_rmse(denorm_pred["pressure"], denorm_targets["pressure"])
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        
        inputs = batch[self.input_variable]  # (B, H, W) or (B, T, H, W)

        
        # Forward pass
        predictions = self(inputs)
        targets = {
            'pressure': batch['pressure'].reshape_as(predictions['pressure']),
        }
        
        
        # Compute loss
        loss = self._compute_loss(predictions, targets)
        
        # Check for NaN in loss
        if torch.isnan(loss):
            print(f"Warning: NaN loss detected in batch {batch_idx}")
            return None
        
        # Update metrics
        self._update_metrics(predictions, targets, "train")
        
        # Log loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('max_pressure', torch.max(batch['pressure']), on_step=True, on_epoch=False, prog_bar=False)
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """Validation step."""
        
        inputs = batch[self.input_variable]
        
        # Forward pass
        predictions = self(inputs)
        
        targets = {
            'pressure': batch['pressure'].reshape_as(predictions['pressure']),
        }
        
        # Compute loss
        loss = self._compute_loss(predictions, targets)
        
        # Update metrics
        self._update_metrics(predictions, targets, "val")
        
        # Store for plotting
        # self.validation_outputs.append({
        #     'predictions': predictions,
        #     'targets': targets,
        #     'cyclone_id': batch['cyclone_id'],
        # })
        
        # Log loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('max_pressure', torch.max(batch['pressure']), on_step=True, on_epoch=False, prog_bar=False)
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        """Test step."""

        inputs = batch[self.input_variable]

        # Forward pass
        predictions = self(inputs)
        targets = {
            'pressure': batch['pressure'].reshape_as(predictions['pressure']),
        }

        # Compute loss
        loss = self._compute_loss(predictions, targets)

        # Update metrics
        self._update_metrics(predictions, targets, "test")

        # Store for plotting and CSV export
        # Denormalize predictions and targets for CSV output
        denorm_pred, denorm_targets = self._denormalize_targets(predictions, targets)

        self.test_outputs.append({
            'predictions': {k: v.detach().cpu() for k, v in predictions.items()},
            'predictions_denorm': {k: v.detach().cpu() for k, v in denorm_pred.items()},
            'targets': {k: v.detach().cpu() for k, v in targets.items()},
            'targets_denorm': {k: v.detach().cpu() for k, v in denorm_targets.items()},
            'cyclone_id': batch['cyclone_id'],
            'frame_idx': batch.get('frame_idx', None),  # Optional frame index tracking
        })

        # Log loss
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        return loss
    
    def on_train_epoch_end(self):
        """Log metrics at end of training epoch."""
        
        self.log("train_pressure_rmse", self.train_pressure_rmse)
        
    
    def on_validation_epoch_end(self):
        """Log metrics and create plots at end of validation epoch."""
        
        self.log("val_pressure_rmse", self.val_pressure_rmse)

        
        # Create scatter plot (optional, can be expensive)
        # if self.current_epoch % 10 == 0 and len(self.validation_outputs) > 0:
        #     self._create_scatter_plot(self.validation_outputs, "val")
        
        # Clear outputs
        self.validation_outputs.clear()
    
    def on_test_epoch_end(self):
        """Log metrics and create plots at end of test epoch."""

        self.log("test_pressure_rmse", self.test_pressure_rmse)

        # Save predictions to CSV if output path is specified
        if self.output_csv_path and len(self.test_outputs) > 0:
            self._save_test_predictions_to_csv()

        # Clear outputs
        self.test_outputs.clear()

    def _save_test_predictions_to_csv(self):
        """Save test predictions to CSV file."""
        rows = []
        global_frame_idx = 0
        cyclone_idx_map = {}  # Map cyclone_id to cyclone_idx

        for batch_output in self.test_outputs:
            cyclone_ids = batch_output['cyclone_id']
            predictions_denorm = batch_output['predictions_denorm']
            targets_denorm = batch_output['targets_denorm']
            frame_indices = batch_output.get('frame_idx', None)

            batch_size = len(cyclone_ids)

            for i in range(batch_size):
                cyclone_id = cyclone_ids[i]

                # Assign cyclone_idx
                if cyclone_id not in cyclone_idx_map:
                    cyclone_idx_map[cyclone_id] = len(cyclone_idx_map)
                cyclone_idx = cyclone_idx_map[cyclone_id]

                # Get frame_idx (use provided or track globally)
                if frame_indices is not None and frame_indices[i] is not None:
                    frame_idx = int(frame_indices[i])
                else:
                    frame_idx = global_frame_idx
                    global_frame_idx += 1

                # Get predictions and targets
                pred_pressure = float(predictions_denorm['pressure'][i].squeeze())
                target_pressure = float(targets_denorm['pressure'][i].squeeze())

                rows.append({
                    'model_family': 'cnn',
                    'model_size': self.model_name,
                    'cyclone_id': cyclone_id,
                    'targets': target_pressure,
                    'y_pred': pred_pressure,
                    'cyclone_idx': cyclone_idx,
                    'frame_idx': frame_idx,
                })

        # Create DataFrame and save
        df = pd.DataFrame(rows)

        # Ensure output directory exists
        output_path = Path(self.output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} test predictions to {output_path}")
    
    
    def configure_optimizers(self):
        """
        Configure optimizer and scheduler - must be implemented by subclasses
        or can use default AdamW.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
