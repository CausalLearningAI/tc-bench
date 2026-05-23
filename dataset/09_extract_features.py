import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset, DatasetDict
from pathlib import Path
import argparse
import json
import os

FRAME_MIN, FRAME_MAX = 140, 375
from transformers import (
    AutoModel, 
    CLIPVisionModel, 
    ViTMAEModel, 
    SiglipVisionModel,
    Siglip2VisionModel
)
model_defaults = {
    # DINO Series
    'dinov2-base': {'model_class': 'auto', 
                    'backbone_name': 'vit', 
                    'model_name': "facebook/dinov2-base"},
    'dinov2-large': {'model_class': 'auto',
                     'backbone_name': 'vit', 
                     'model_name': "facebook/dinov2-large"},
    'dinov3-base': {'model_class': 'auto',
                    'backbone_name': 'vit', 
                    'model_name': "facebook/dinov3-vitb16-pretrain-lvd1689m"},
    'dinov3-large': {'model_class': 'auto',
                     'backbone_name': 'vit', 
                     'model_name': "facebook/dinov3-vitl16-pretrain-lvd1689m"},
    'dinov3-satellite': {'model_class': 'auto',
                         'backbone_name': 'vit', 
                         'model_name': "facebook/dinov3-vitl16-pretrain-sat493m"},
    # CLIP & SigLIP
    'clip-base': {'model_class': 'clip', 
                  'backbone_name': 'clip', 
                  'model_name': 'openai/clip-vit-base-patch16'},
    'clip-large': {'model_class': 'clip',
                   'backbone_name': 'clip', 
                   'model_name': 'openai/clip-vit-large-patch14'},
    'siglip-base': {'model_class': 'siglip', 
                    'backbone_name': 'siglip', 
                    'model_name': 'google/siglip-base-patch16-224'},
    # 'siglip-large': {'model_class': 'siglip', 
    #                  'backbone_name': 'siglip', 
    #                  'model_name': 'google/siglip-so400m-patch14-384'},
    'siglip2-base': {'model_class': 'siglip', 
                     'backbone_name': 'siglip', 
                     'model_name': 'google/siglip2-base-patch16-224'},
    # 'siglip2-large': {'model_class': 'siglip2', 
    #                   'backbone_name': 'siglip', 
    #                   'model_name': 'google/siglip2-so400m-patch14-384'},
    # 'siglip2-naflex': {'model_class': 'siglip2', 
    #                   'backbone_name': 'siglip', 
    #                   'model_name': 'google/siglip2-so400m-patch16-naflex'},

    # Generative / Masked Autoencoders
    'mae-base': {'model_class': 'mae', 
                 'backbone_name': 'mae', 
                 'model_name': 'facebook/vit-mae-base'},
    'mae-large': {'model_class': 'mae', 
                  'backbone_name': 'mae', 
                  'model_name': 'facebook/vit-mae-large'},
    
    # Clay Series (Multimodal/Temporal Stacks)
    'clay-base': {'model_class': 'auto', 
                  'backbone_name': 'clay', 
                  'model_name': 'made-with-clay/Clay'},
    'clay-legacy': {'model_class': 'auto',
                    'backbone_name': 'clay', 
                    'model_name': 'made-with-clay/Clay-legacy'},

}
    
video_models = {
    # Prithvi Series (IBM-NASA Geospatial)
    'prithvi-100m': {'backbone_name': 'prithvi', 'model_name': "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"},
    'prithvi-600m': {'backbone_name': 'prithvi', 'model_name': "ibm-nasa-geospatial/Prithvi-EO-2.0-600M"},
    'prithvi-wxc-2.3b': {'backbone_name': 'prithvi-wxc', 'model_name': "ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M"}
}
    

class FeatureExtractor(nn.Module):
    def __init__(self, model_type:str):
        super().__init__()
        self.model_type = model_type
        model_config = model_defaults.get(model_type) or video_models.get(model_type)
        self.model_class = model_config['model_class']
        self.backbone_name = model_config['backbone_name']
        self.model_name = model_config['model_name']
        self.frame_min = FRAME_MIN
        self.frame_max = FRAME_MAX
        
        # Load backbone
        self.backbone = self._prepare_backbone()
        self.backbone.eval()
        
        # Check for DINOv3 registers
        self.num_registers = getattr(self.backbone.config, "num_register_tokens", 0)

    def _prepare_backbone(self):
        """Standardizes loading based on model_class."""
        if self.model_class == 'auto':
            if "clay" in self.model_name.lower():
                # Use 'main' revision to bypass older, broken snapshots
                return AutoModel.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True, 
                    use_safetensors=True,
                    revision="main" 
                )
            else:
                # Clay models need trust_remote_code
                return AutoModel.from_pretrained(self.model_name, trust_remote_code=True, use_safetensors=True)
        elif self.model_class == 'clip':
            return CLIPVisionModel.from_pretrained(self.model_name, use_safetensors=True)
        elif self.model_class == 'siglip':
            return SiglipVisionModel.from_pretrained(self.model_name, use_safetensors=True)
        elif self.model_class == 'siglip2':
            # Fallback to AutoModel if Siglip2VisionModel isn't in your transformers version yet
            return Siglip2VisionModel.from_pretrained(self.model_name, use_safetensors=True)
        elif self.model_class == 'mae':
            return ViTMAEModel.from_pretrained(self.model_name, use_safetensors=True)
        raise ValueError(f"Unknown model class: {self.model_class}")

    def forward(self, x: torch.Tensor) -> dict:
        """Processes (B, 1, H, W) and returns Tensor features."""
        # 1. Normalization & RGB Replication
        print(f'Normalizing frames with min={self.frame_min} and max={self.frame_max}')
        x = (x - self.frame_min) / (self.frame_max - self.frame_min)
        x = torch.clamp(x, 0, 1).repeat(1, 3, 1, 1)

        with torch.no_grad():
            outputs = self.backbone(x)
            
            # Extract based on architecture 'backbone_name'
            if self.backbone_name == 'vit':
                # Shape: [B, 1 + num_reg + num_patches, D]
                hidden_states = outputs.last_hidden_state
                cls_features = hidden_states[:, 0]
                # Skip CLS (0) and any registers (1:1+reg)
                spatial_mean = hidden_states[:, 1 + self.num_registers:].mean(dim=1)

            elif self.backbone_name == 'mae':
                # MAE architecture (usually matches ViT topology)
                hidden_states = outputs.last_hidden_state
                cls_features = hidden_states[:, 0]
                spatial_mean = hidden_states[:, 1:].mean(dim=1)

            elif self.backbone_name == 'siglip':
                # SigLIP uses global pooling (pooler_output)
                cls_features = outputs.pooler_output
                spatial_mean = outputs.last_hidden_state.mean(dim=1)

            elif self.backbone_name == 'clip':
                # CLIP pooler_output is the projection, CLS is the raw token
                cls_features = outputs.pooler_output
                spatial_mean = outputs.last_hidden_state[:, 1:].mean(dim=1)
            
            else:
                raise ValueError(f"Feature mapping not defined for: {self.backbone_name}")

        return {"cls": cls_features, "spatial_mean": spatial_mean}

def extract_features_for_split(extractor, dataset, split_name, batch_size=32, num_workers=4, device='cuda'):
    print(f"\n🚀 Extraction: {split_name} ({len(dataset)} cyclones)")
    extractor.to(device).eval()
    
    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers, collate_fn=lambda x: x)
    results = []
    
    for batch in tqdm(dataloader, desc=split_name):
        ex = batch[0]
        frames = np.array(ex['frames'])
        frame_valid_mask = np.array(ex['frame_valid_mask'], dtype=bool)
        # fill-in invalid frames with zeros
        mask = frame_valid_mask.copy()
        frames[~frame_valid_mask] = np.zeros_like(frames[~frame_valid_mask])
        T = frames.shape[0]
        
        cls_acc, spatial_acc = [], []

        for i in range(0, T, batch_size):
            b_frames = torch.from_numpy(frames[i:i+batch_size]).unsqueeze(1).float().to(device)
            
            with torch.no_grad():
                out = extractor(b_frames)
            
            # Convert to CPU/Numpy only once per batch
            cls_acc.append(out['cls'].cpu().numpy())
            spatial_acc.append(out['spatial_mean'].cpu().numpy())

        # Store results back to the standard format
        results.append({
            **{k: v for k, v in ex.items() if k != 'frames'}, # Keep all metadata
            'features': {
                'cls': np.concatenate(cls_acc, axis=0).tolist(),
                'spatial_mean': np.concatenate(spatial_acc, axis=0).tolist(),
            },
            'frame_valid_mask': mask.tolist(),
            'num_timesteps': T,
            'feature_dim': cls_acc[0].shape[1]
        })
    return results
    
def main():
    parser = argparse.ArgumentParser(description="Extract features from vision models")
    parser.add_argument(
        '--model_type',
        type=str,
        required=True,
        choices=list(model_defaults.keys()) + list(video_models.keys()),
        help='Type of model to use'
    )
    data_root = os.environ.get("DATA_ROOT", os.path.join(os.environ["HOME"], "tcbench"))
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=os.path.join(data_root, 'dataset_hf'),
        help='Path to HuggingFace dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.path.join(data_root, 'image_features'),
        help='Output directory for feature datasets'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size for feature extraction (frames per GPU batch)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of DataLoader workers for prefetching'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )
    
    args = parser.parse_args()

    # Get model config from defaults
    model_config = model_defaults.get(args.model_type) or video_models.get(args.model_type)
    backbone_name = model_config['backbone_name']
    model_name = model_config['model_name']

    print("="*80)
    print(f"Extracting features with {args.model_type.upper()}")
    print("="*80)
    print(f"Backbone type: {backbone_name}")
    print(f"Model: {model_name}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num workers: {args.num_workers}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset_dict = load_from_disk(args.dataset_path)
    print(f"✓ Loaded dataset with {len(dataset_dict['train'])} train, "
          f"{len(dataset_dict['validation'])} val, {len(dataset_dict['test'])} test cyclones")
    
    # Load normalization statistics
    # stats_path = Path(args.dataset_path) / 'normalization_stats.json'
    # if stats_path.exists():
    #     with open(stats_path) as f:
    #         stats = json.load(f)
    #     frame_min = stats['frame_min']
    #     frame_max = stats['frame_max']
    #     print(f"\n✓ Loaded normalization stats from {stats_path}")
    #     print(f"  Frame range: [{frame_min}, {frame_max}] K")
    # else:
    #     print(f"\n⚠ Warning: normalization_stats.json not found at {stats_path}")
    #     print(f"  Using default frame range: [140, 375] K")
    #     frame_min, frame_max = 140, 375
     # TODO: load from valid range
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"features_{args.model_type}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving feature dataset to: {output_dir}")
    
    # Initialize extractor
    print(f"\nLoading {args.model_type.upper()} model...")
    extractor = FeatureExtractor(args.model_type)
    print(f"✓ Model loaded")
    
    
    # Extract features for each split
    feature_datasets = {}
    for split_name in ['train', 'validation', 'test']:
        results = extract_features_for_split(
            extractor=extractor,
            dataset=dataset_dict[split_name],
            split_name=split_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device
        )
        
        # Convert to HuggingFace Dataset
        feature_datasets[split_name] = Dataset.from_list(results)
    
    # Create DatasetDict
    feature_dataset_dict = DatasetDict(feature_datasets)
    
    # Save to disk
    print(f"\nSaving feature dataset...")
    feature_dataset_dict.save_to_disk(str(output_dir))
    
    # Save model info
    info_path = output_dir / "model_info.txt"
    with open(info_path, 'w') as f:
        f.write(f"Model Type: {args.model_type}\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Source Dataset: {args.dataset_path}\n")
        f.write(f"\nDataset Structure:\n{feature_dataset_dict}\n")
    
    print("\n" + "="*80)
    print(f"✓ Feature extraction complete!")
    print(f"✓ Saved to: {output_dir}")
    print("="*80)
    
#     video_models = {
#     # Scale4D / 4DS Models (The Geometry-first Web-Video models)
#     'scale4d-base':  {'backbone': 'video',   'model_name': 'google-deepmind/4ds-base'},
#     'scale4d-large': {'backbone': 'video',   'model_name': 'google-deepmind/4ds-large'},
    
#     # Traditional Video Models (The Motion-first Web-Video models)
#     'videomae-base': {'backbone': 'video',   'model_name': "MCG-NJU/videomae-base-finetuned-kinetics"},
#     'vjepa-base':    {'backbone': 'video',   'model_name': "facebook/vjepa-vit-base-p16"},
# }


if __name__ == '__main__':
    main()
