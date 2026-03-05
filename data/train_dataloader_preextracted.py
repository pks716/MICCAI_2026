"""
Fast Dataloader for Pre-extracted Patches

Loads pre-extracted .npy patches from disk (created by extract_patches.py)
This is 10-20× faster than loading original .nii.gz files!
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json


class PreExtractedPatchDataset(Dataset):
    """
    Assumes patches have been extracted using extract_patches.py
    
    Structure:
        output_dir/
            patches/
                patch_000000_ct.npy
                patch_000000_mr.npy
                patch_000000_mask.npy
                patch_000001_ct.npy
                ...
            patch_index.json
            metadata.json
    """
    
    def __init__(self, patches_dir, type='CT2MR', patient_list=None):
        """
        Args:
            patches_dir: Directory containing extracted patches
            type: 'CT2MR' or 'MR2CT'
            patient_list: Optional list of patient names to include (for train/val split)
        """
        self.patches_dir = patches_dir
        self.type = type
        
        # Load index
        index_file = os.path.join(patches_dir, 'patch_index.json')
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        with open(index_file, 'r') as f:
            self.patch_index = json.load(f)
        
        # Load metadata
        metadata_file = os.path.join(patches_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Filter by patient list if provided
        if patient_list is not None:
            self.patch_index = [
                p for p in self.patch_index 
                if p['patient_name'] in patient_list
            ]
        
        self.patches_subdir = os.path.join(patches_dir, 'patches')
        
        print(f"✓ Loaded {len(self.patch_index)} pre-extracted patches")
        if patient_list is not None:
            unique_patients = len(set(p['patient_name'] for p in self.patch_index))
            print(f"✓ From {unique_patients} patients")
    
    def __len__(self):
        return len(self.patch_index)
    
    def __getitem__(self, idx):
        """
        Load pre-extracted patch from .npz file
        
        Returns:
            ct_patch: [1, D, H, W]
            mr_patch: [1, D, H, W]
            mask_patch: [1, D, H, W]
        """
        patch_info = self.patch_index[idx]
        
        patch_data = np.load(os.path.join(self.patches_subdir, patch_info['file']))
        
        # Extract arrays
        ct_patch = patch_data['ct']
        mr_patch = patch_data['mr']
        mask_patch = patch_data['mask']
        
        # Convert to tensors
        ct_patch = torch.from_numpy(ct_patch).unsqueeze(0).float()
        mr_patch = torch.from_numpy(mr_patch).unsqueeze(0).float()
        mask_patch = torch.from_numpy(mask_patch).unsqueeze(0).float()
        
        if self.type == 'CT2MR':
            return ct_patch, mr_patch, mask_patch
        elif self.type == 'MR2CT':
            return mr_patch, ct_patch, mask_patch
        else:
            raise ValueError("type must be 'CT2MR' or 'MR2CT'")
    
    def get_patient_names(self):
        """Get list of unique patient names in dataset"""
        return sorted(set(p['patient_name'] for p in self.patch_index))
    
    def get_metadata(self):
        """Get dataset metadata"""
        return self.metadata


def create_train_val_split(patches_dir, val_split=0.2, seed=42):
    """
    Create train/val split based on patients (not patches)
    
    Args:
        patches_dir: Directory with extracted patches
        val_split: Fraction for validation (default: 0.2)
        seed: Random seed
    
    Returns:
        train_patients: List of patient names for training
        val_patients: List of patient names for validation
    """
    # Load full dataset to get patient names
    full_dataset = PreExtractedPatchDataset(patches_dir)
    all_patients = full_dataset.get_patient_names()
    
    # Shuffle and split
    np.random.seed(seed)
    shuffled = np.random.permutation(all_patients)
    
    val_size = int(len(shuffled) * val_split)
    val_patients = shuffled[:val_size].tolist()
    train_patients = shuffled[val_size:].tolist()
    
    print(f"✓ Train/val split:")
    print(f"  Training: {len(train_patients)} patients")
    print(f"  Validation: {len(val_patients)} patients")
    
    return train_patients, val_patients


# Helper function for testing
def test_preextracted_dataloader():
    """Test the pre-extracted patch dataloader"""
    import time
    
    # Configuration
    patches_dir = "/path/to/extracted/patches"
    batch_size = 4
    
    print("="*60)
    print("TESTING PRE-EXTRACTED PATCH DATALOADER")
    print("="*60)
    
    # Create train/val split
    train_patients, val_patients = create_train_val_split(
        patches_dir, 
        val_split=0.2, 
        seed=42
    )
    
    # Create training dataset
    print("\nCreating training dataset...")
    train_dataset = PreExtractedPatchDataset(
        patches_dir=patches_dir,
        type='CT2MR',
        patient_list=train_patients
    )
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Can use more workers since loading is fast
        pin_memory=True,
        persistent_workers=True,
    )
    
    # Test speed
    print("\nTesting loading speed...")
    times = []
    
    for i, (ct_patch, mr_patch, mask_patch) in enumerate(train_loader):
        start = time.time()
        
        # Simulate GPU transfer
        if torch.cuda.is_available():
            ct_patch = ct_patch.cuda()
            mr_patch = mr_patch.cuda()
            mask_patch = mask_patch.cuda()
        
        times.append(time.time() - start)
        
        if i == 0:
            print(f"\nBatch shapes:")
            print(f"  CT: {ct_patch.shape}")
            print(f"  MR: {mr_patch.shape}")
            print(f"  Mask: {mask_patch.shape}")
            print(f"  CT range: [{ct_patch.min():.3f}, {ct_patch.max():.3f}]")
            print(f"  MR range: [{mr_patch.min():.3f}, {mr_patch.max():.3f}]")
        
        if i >= 20:  # Test 20 batches
            break
    
    # Skip first few for warmup
    avg_time = np.mean(times[5:])
    
    print(f"\nPerformance:")
    print(f"  Average batch time: {avg_time:.4f}s")
    print(f"  Batches per second: {1.0/avg_time:.1f}")
    print(f"  Samples per second: {batch_size/avg_time:.1f}")
    
    print("\n✓ Pre-extracted dataloader working perfectly!")
    print("="*60)
    
    # Create validation dataset for comparison
    print("\nCreating validation dataset...")
    val_dataset = PreExtractedPatchDataset(
        patches_dir=patches_dir,
        type='CT2MR',
        patient_list=val_patients
    )
    
    print(f"✓ Validation dataset: {len(val_dataset)} patches")
    
    return train_loader


if __name__ == "__main__":
    test_preextracted_dataloader()