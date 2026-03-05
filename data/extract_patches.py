import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm
import json
import argparse
from monai.transforms import CropForeground


def find_file(patient_folder, modality):
    extensions = ['.nii.gz', '.nii', '.mha']
    name_patterns = [modality, f'{modality}_iso']
    for name in name_patterns:
        for ext in extensions:
            filepath = os.path.join(patient_folder, f'{name}{ext}')
            if os.path.exists(filepath):
                return filepath
    return None


def load_volume(filepath):
    if filepath.endswith('.mha'):
        img = sitk.ReadImage(filepath)
        volume = sitk.GetArrayFromImage(img)
    else:
        img = nib.load(filepath)
        volume = img.get_fdata()
        volume = np.transpose(volume, (2, 0, 1))
    return volume.astype(np.float32)


def preprocess_ct(ct_volume, mask_volume):
    """Clip HU then min-max to [0,1] and mask."""
    ct_volume = np.clip(ct_volume, -1024, 3000)
    ct_min = ct_volume.min()
    ct_max = ct_volume.max()
    if ct_max > ct_min:
        ct_normalized = (ct_volume - ct_min) / (ct_max - ct_min)
    else:
        ct_normalized = np.zeros_like(ct_volume)
    mask_binary = (mask_volume > 0.5).astype(np.float32)
    return ct_normalized * mask_binary


def preprocess_mr(mr_volume, mask_volume):
    """Min-max to [0,1] and mask."""
    mr_min = mr_volume.min()
    mr_max = mr_volume.max()
    if mr_max > mr_min:
        mr_normalized = (mr_volume - mr_min) / (mr_max - mr_min)
    else:
        mr_normalized = np.zeros_like(mr_volume)
    mask_binary = (mask_volume > 0.5).astype(np.float32)
    return mr_normalized * mask_binary


def crop_foreground(ct_volume, mr_volume, mask_volume):
    """Crop all volumes to foreground bounding box using mask."""
    import torch
    m = torch.from_numpy(mask_volume[np.newaxis]).float()  # [1, D, H, W]
    cropper = CropForeground(select_fn=lambda x: x > 0.5)
    box_start, box_end = cropper.compute_bounding_box(m)
    box_start = [int(v) for v in box_start]
    box_end   = [int(v) for v in box_end]

    ct_cropped   = ct_volume  [box_start[0]:box_end[0], box_start[1]:box_end[1], box_start[2]:box_end[2]]
    mr_cropped   = mr_volume  [box_start[0]:box_end[0], box_start[1]:box_end[1], box_start[2]:box_end[2]]
    mask_cropped = mask_volume[box_start[0]:box_end[0], box_start[1]:box_end[1], box_start[2]:box_end[2]]
    return ct_cropped, mr_cropped, mask_cropped

def extract_grid_patches(ct_volume, mr_volume, mask_volume, patch_size):
    D, H, W = ct_volume.shape
    ps = patch_size

    # Pad if any dimension is smaller than patch_size
    if D < ps or H < ps or W < ps:
        ct_volume   = np.pad(ct_volume,   [(0, max(0, ps-D)), (0, max(0, ps-H)), (0, max(0, ps-W))])
        mr_volume   = np.pad(mr_volume,   [(0, max(0, ps-D)), (0, max(0, ps-H)), (0, max(0, ps-W))])
        mask_volume = np.pad(mask_volume, [(0, max(0, ps-D)), (0, max(0, ps-H)), (0, max(0, ps-W))])
        D, H, W = ct_volume.shape
    patches = []

    for d in range(0, D, ps):
        for h in range(0, H, ps):
            for w in range(0, W, ps):
                # Clamp to volume bounds
                d_end = min(d + ps, D)
                h_end = min(h + ps, H)
                w_end = min(w + ps, W)
                d_start = d_end - ps
                h_start = h_end - ps
                w_start = w_end - ps

                mask_patch = mask_volume[d_start:d_end, h_start:h_end, w_start:w_end]
                if (mask_patch > 0.5).any():  # skip empty patches
                    ct_patch   = ct_volume  [d_start:d_end, h_start:h_end, w_start:w_end]
                    mr_patch   = mr_volume  [d_start:d_end, h_start:h_end, w_start:w_end]
                    patches.append((ct_patch, mr_patch, mask_patch))

    return patches


def extract_patches_from_dataset(data_dir, output_dir, patch_size=96,
                                 patches_per_volume=3, seed=42):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    patches_dir = os.path.join(output_dir, 'patches')
    os.makedirs(patches_dir, exist_ok=True)

    print("="*60)
    print("EXTRACTING PATCHES TO DISK")
    print("="*60)
    print(f"Source: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Patch size: {patch_size}³")
    print(f"Patches per volume: {patches_per_volume}")
    print("="*60 + "\n")

    patient_folders = sorted([
        os.path.join(data_dir, p)
        for p in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, p)) and not p.startswith('.')
    ])

    patch_index = []
    patch_id = 0

    for patient_folder in tqdm(patient_folders, desc="Processing patients"):
        patient_name = os.path.basename(patient_folder)
        try:
            ct_path   = find_file(patient_folder, 'ct')
            mr_path   = find_file(patient_folder, 'mr')
            mask_path = find_file(patient_folder, 'mask')

            if not (ct_path and mr_path and mask_path):
                tqdm.write(f"⚠️  Skipping {patient_name}: missing files")
                continue

            ct_volume   = load_volume(ct_path)
            mr_volume   = load_volume(mr_path)
            mask_volume = load_volume(mask_path)

            # Preprocess
            ct_preprocessed   = preprocess_ct(ct_volume, mask_volume)
            mr_preprocessed   = preprocess_mr(mr_volume, mask_volume)

            # Crop foreground
            ct_preprocessed, mr_preprocessed, mask_volume = crop_foreground(
                ct_preprocessed, mr_preprocessed, mask_volume
            )

            for patch_num, (ct_patch, mr_patch, mask_patch) in enumerate(
                extract_grid_patches(ct_preprocessed, mr_preprocessed, mask_volume, patch_size)):

                patch_filename = f"patch_{patch_id:06d}.npz"
                np.savez_compressed(
                    os.path.join(patches_dir, patch_filename),
                    ct=ct_patch, mr=mr_patch, mask=mask_patch
                )

                patch_index.append({
                    'patch_id': patch_id,
                    'patient_name': patient_name,
                    'patch_num': patch_num,
                    'file': patch_filename,
                    'shape': list(ct_patch.shape)
                })
                patch_id += 1

        except Exception as e:
            tqdm.write(f"  Error processing {patient_name}: {e}")
            continue

    index_file = os.path.join(output_dir, 'patch_index.json')
    with open(index_file, 'w') as f:
        json.dump(patch_index, f, indent=2)

    metadata = {
        'patch_size': patch_size,
        'patches_per_volume': patches_per_volume,
        'total_patches': len(patch_index),
        'total_patients': len(set(p['patient_name'] for p in patch_index)),
        'seed': seed,
        'data_dir': data_dir
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Total patches: {len(patch_index)}")
    print(f"✓ Total patients: {metadata['total_patients']}")
    print(f"✓ Index saved: {index_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',           type=str, required=True)
    parser.add_argument('--output_dir',         type=str, required=True)
    parser.add_argument('--patch_size',         type=int, default=96)
    parser.add_argument('--patches_per_volume', type=int, default=4)
    parser.add_argument('--seed',               type=int, default=42)
    args = parser.parse_args()

    extract_patches_from_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        patches_per_volume=args.patches_per_volume,
        seed=args.seed
    )