import os
import shutil
import json
import argparse
import numpy as np
from tqdm import tqdm


def check_patient_files(patient_folder):
    extensions = ['.nii.gz', '.nii', '.mha']
    modalities = ['ct', 'mr', 'mask']
    name_patterns = {
        'ct':   ['ct', 'ct_iso'],
        'mr':   ['mr', 'mr_iso'],
        'mask': ['mask', 'mask_iso']
    }
    for modality in modalities:
        found = any(
            os.path.exists(os.path.join(patient_folder, f'{name}{ext}'))
            for name in name_patterns[modality]
            for ext in extensions
        )
        if not found:
            return False, f"Missing {modality}"
    return True, "OK"


def create_split_folders(data_dir, output_dir,
                         train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
                         mode='symlink', seed=42):
    np.random.seed(seed)

    print("="*60)
    print(f"Source : {data_dir}")
    print(f"Output : {output_dir}")
    print(f"Split  : Train {train_ratio*100:.0f}% | Val {val_ratio*100:.0f}% | Test {test_ratio*100:.0f}%")
    print(f"Mode   : {mode.upper()}")
    print("="*60 + "\n")

    # Gather valid patients
    patients = []
    for p in sorted(os.listdir(data_dir)):
        folder = os.path.join(data_dir, p)
        if not os.path.isdir(folder) or p.startswith('.'):
            continue
        ok, _ = check_patient_files(folder)
        if ok:
            patients.append(folder)

    n = len(patients)
    print(f"Found {n} valid patients\n")

    shuffled = np.random.permutation(patients).tolist()
    n_train  = int(n * train_ratio)
    n_val    = int(n * val_ratio)

    splits = {
        'train': shuffled[:n_train],
        'val':   shuffled[n_train:n_train + n_val],
        'test':  shuffled[n_train + n_val:]
    }

    print(f"Train: {len(splits['train'])}  Val: {len(splits['val'])}  Test: {len(splits['test'])}\n")

    def link_patient(src, dst):
        if mode == 'symlink':
            os.symlink(os.path.abspath(src), dst, target_is_directory=True)
        else:
            shutil.copytree(src, dst, symlinks=True)

    for split_name, split_patients in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for patient_folder in tqdm(split_patients, desc=split_name):
            patient_name = os.path.basename(patient_folder)
            dst = os.path.join(split_dir, patient_name)
            if os.path.exists(dst):
                continue
            try:
                link_patient(patient_folder, dst)
            except Exception as e:
                print(f" {patient_name}: {e}")

    # Save split info
    info = {
        'data_dir': data_dir,
        'output_dir': output_dir,
        'seed': seed,
        'counts': {k: len(v) for k, v in splits.items()}
    }
    with open(os.path.join(output_dir, 'split_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n✓ Done. Split info → {output_dir}/split_info.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    type=str, required=True)
    parser.add_argument('--output_dir',  type=str, required=True)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio',   type=float, default=0.1)
    parser.add_argument('--test_ratio',  type=float, default=0.2)
    parser.add_argument('--mode',        type=str, default='symlink', choices=['copy', 'symlink'])
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.001:
        raise ValueError("Ratios must sum to 1.0")

    create_split_folders(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        mode=args.mode,
        seed=args.seed
    )