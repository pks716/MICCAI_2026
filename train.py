import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import os
import pandas as pd
from collections import deque
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchvision.utils import make_grid, save_image
import json
import pickle
import torch.nn as nn
import random
from flow_matching.solver import ODESolver
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler


# ============ IMPORT 3D COMPONENTS ============
from data.train_dataloader_preextracted import PreExtractedPatchDataset
from models.utils_fm import build_model 
from models.model_3d import Model as Model3D
from models.deform import DeformationSampler3D, apply_deformation_3d

from config import Config

path = AffineProbPath(scheduler=CondOTScheduler()) 

HP = Config()

if HP.USE_WANDB:
    import wandb

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(HP.RANDOM_SEED)

# ============ METRICS ============
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(HP.DEVICE)
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(HP.DEVICE)

# ============ FLOW MATCHING ============

class OptimalTransportConditionalFlowMatcher:
    def __init__(self, sigma=0.0):
        self.sigma = sigma
    
    def sample_location_and_conditional_flow(self, x0, x1, t=None):
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device)
        t = t.view(-1, 1, 1, 1, 1)
        mu_t = (1 - t) * x0 + t * x1
        x_t = mu_t + torch.randn_like(x0) * self.sigma if self.sigma > 0 else mu_t
        u_t = x1 - x0
        return x_t, u_t

flow_matcher = OptimalTransportConditionalFlowMatcher(sigma=0.0)

def flow_matching_loss(model, source_patch, target_patch):
    batch_size = source_patch.shape[0]
    t = torch.rand(batch_size, device=source_patch.device)
    sample_info = path.sample(t=t, x_0=source_patch, x_1=target_patch)
    v_pred = model(torch.cat([source_patch, sample_info.x_t], dim=1), sample_info.t)
    return F.mse_loss(v_pred, sample_info.dx_t)


@torch.no_grad()
def sample_flow_matching_3d(model, source_patch, steps=2, method='euler', device='cuda'):
    model.eval()
    B = source_patch.shape[0]
    
    class ConditionedModel(nn.Module):
        def __init__(self, base_model, src):
            super().__init__()
            self.base_model = base_model
            self.src        = src  # explicitly stored, not closure
        
        def forward(self, x, t):
            t_batch = t.expand(B) if t.dim() == 0 else t
            return self.base_model(torch.cat([self.src, x], dim=1), t_batch)
    
    conditioned = ConditionedModel(model, source_patch)
    solver = ODESolver(velocity_model=conditioned)
    T = torch.linspace(0, 1, steps + 1, device=device)
    sol = solver.sample(
        time_grid = T,
        x_init = source_patch.clone(),
        method = method,
        step_size = 1.0 / steps,
        return_intermediates = False,
    )
    return sol


@torch.no_grad()
def sample_flow_matching_ensemble_3d(model, deformation_sampler, source_patch,
                                     n_samples=3, steps=5, method='euler', device='cuda'):
    model.eval()
    deformation_sampler.eval()
    synth_samples = []

    for _ in range(n_samples):
        deformation, _, _ = deformation_sampler(source_patch, sample=True)
        source_warped     = apply_deformation_3d(source_patch, deformation)
        synth = sample_flow_matching_3d(
        model, source_warped,
        steps=steps, method=method, device=device
    )
        synth_samples.append(synth)

    synth_samples = torch.stack(synth_samples, dim=0)   # [N, B, 1, D, H, W]
    ensemble = torch.mean(synth_samples, dim=0)
    uncertainty = torch.std(synth_samples,  dim=0)
    return ensemble, synth_samples, uncertainty


# ============ EMA HELPER ============
class EMAHelper:
    def __init__(self, mu=0.9999):
        self.mu     = mu
        self.shadow = {}
    
    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (self.mu * self.shadow[name].data +
                                          (1.0 - self.mu) * param.data)
    
    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)
    
    def store(self, parameters):
        self.backup = {}
        for name, param in parameters:
            if param.requires_grad:
                self.backup[name] = param.data.clone()
    
    def restore(self, parameters):
        for name, param in parameters:
            if param.requires_grad:
                param.data.copy_(self.backup[name])
    
    def state_dict(self):          return self.shadow
    def load_state_dict(self, sd): self.shadow = sd


# ============ CHECKPOINT ============
def save_checkpoint(model, optimizer, iteration, path,
                    ema_helper=None, deformation_sampler=None, scaler=None, scheduler=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    states = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    if ema_helper is not None: states['ema_state_dict'] = ema_helper.state_dict()
    if deformation_sampler is not None: states['deformation_sampler_state_dict'] = deformation_sampler.state_dict()
    if scaler is not None: states['scaler_state_dict'] = scaler.state_dict()
    # After scaler:
    if scheduler is not None: states['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(states, path)


def load_checkpoint(model, optimizer, checkpoint, device=None,
                    ema_helper=None, deformation_sampler=None, scheduler=None):
    if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
    if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if ema_helper is not None and 'ema_state_dict' in checkpoint:
        ema_helper.load_state_dict(checkpoint['ema_state_dict'])
    if deformation_sampler is not None and 'deformation_sampler_state_dict' in checkpoint:
        deformation_sampler.load_state_dict(checkpoint['deformation_sampler_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint.get('iteration', 0)


# ============ VISUALIZATION ============
def save_middle_slice_visualization(source_patch, target_real, target_fake,
                                    mask_patch, save_path, iteration):
    """Shows: [source | target_real | target_fake] — works for both MR2CT and CT2MR."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    B, C, D, H, W = source_patch.shape
    mid = D // 2

    src_sl = source_patch[:, 0, mid, :, :]
    real_sl = target_real [:, 0, mid, :, :]
    fake_sl = target_fake [:, 0, mid, :, :]
    mask_sl = mask_patch  [:, 0, mid, :, :]

    real_sl = real_sl * mask_sl
    fake_sl = fake_sl * mask_sl

    rows = []
    for i in range(min(B, 4)):
        row = torch.cat([
            src_sl [i:i+1].unsqueeze(0),
            real_sl[i:i+1].unsqueeze(0),
            fake_sl[i:i+1].unsqueeze(0)
        ], dim=3)
        rows.append(row)

    if rows:
        save_image(torch.cat(rows, dim=2), save_path)


# ============ METRICS ============
def compute_3d_metrics(pred_volume, target_volume, mask_volume):
    pred_tensor = torch.from_numpy(pred_volume  ).unsqueeze(0).unsqueeze(0).float().to(HP.DEVICE)
    target_tensor = torch.from_numpy(target_volume).unsqueeze(0).unsqueeze(0).float().to(HP.DEVICE)
    mask_tensor = torch.from_numpy(mask_volume  ).unsqueeze(0).unsqueeze(0).float().to(HP.DEVICE)
    
    mask_binary = (mask_tensor > 0.5).float()
    pred_masked = pred_tensor   * mask_binary
    target_masked = target_tensor * mask_binary
    
    psnr_values, ssim_values = [], []
    for d in range(pred_volume.shape[0]):
        pred_slice = pred_masked  [0, 0, d:d+1, :, :].unsqueeze(0)
        target_slice = target_masked[0, 0, d:d+1, :, :].unsqueeze(0)
        if target_slice.sum() > 0:
            psnr_values.append(psnr_metric(pred_slice, target_slice).item())
            ssim_values.append(ssim_metric(pred_slice, target_slice).item())

    return {
        'psnr': np.mean(psnr_values) if psnr_values else 0.0,
        'ssim': np.mean(ssim_values) if ssim_values else 0.0,
        'mse':  F.mse_loss(pred_masked, target_masked).item(),
        'mae':  F.l1_loss (pred_masked, target_masked).item(),
    }


# ============ MAIN TRAINING ============
def main():
    
    base_path = f"sessions/{HP.EXPERIMENT_NAME}"
    os.makedirs(base_path, exist_ok=True)
    with open(f"{base_path}/config.json", 'w') as f:
        json.dump(vars(HP), f, indent=2)
    
    # ── Dataset ──────────────────────────────────────────────────────────────
    train_dataset = PreExtractedPatchDataset(patches_dir=HP.TRAIN_PATCHES_DIR, type=HP.TASK)
    val_dataset = PreExtractedPatchDataset(patches_dir=HP.VAL_PATCHES_DIR, type=HP.TASK)
    print(f"✓ Training patches: {len(train_dataset)}")
    print(f"✓ Validation patches: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=HP.BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=HP.BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    print(f"✓ Train batches per epoch: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    small_config = {
        "spatial_dims": 3,
        "in_channels": 2,           # source_x_t + target (conditioning)
        "out_channels": 1,          # velocity field
        "num_res_blocks": [2, 2, 2, 2],
        "num_channels": [32, 64, 128, 256],
        "attention_levels": [False, False, False, True],
        "norm_num_groups": 32,
        "resblock_updown": True,
        "num_head_channels": [32, 64, 128, 256],
        "transformer_num_layers": 6,
        "use_flash_attention": True,
        "with_conditioning": False,
        "mask_conditioning": False
    }

    # model = build_model(small_config, device=torch.device(HP.DEVICE)) #Smaller Capacity Model

    from config import FlowMatchingConfig3D #Bigger Capacity Model
    config = FlowMatchingConfig3D()
    model = Model3D(config).to(HP.DEVICE)

    deformation_sampler = DeformationSampler3D(
        input_channels=1, patch_size=HP.PATCH_SIZE
    ).to(HP.DEVICE)

    model_params = sum(p.numel() for p in model.parameters())
    deform_params = sum(p.numel() for p in deformation_sampler.parameters())
    print(f"✓ Model: {model_params/1e6:.2f}M parameters")
    print(f"✓ Deformation model: {deform_params/1e6:.2f}M parameters")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    all_params = list(model.parameters()) + list(deformation_sampler.parameters())
    optimizer  = optim.AdamW(all_params, lr=HP.LEARNING_RATE, weight_decay=1e-4)
    ema_helper = EMAHelper(mu=0.9999)
    ema_helper.register(model)
    scaler = torch.cuda.amp.GradScaler() if HP.USE_AMP else None
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=HP.LEARNING_RATE,
        total_steps=HP.N_ITERS // HP.GRAD_ACCUM_STEPS,
        pct_start=0.05,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100,
    )



    print(f"✓ Optimizer: AdamW (lr={HP.LEARNING_RATE})")
    print(f"✓ Mixed precision: {HP.USE_AMP}")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_iteration = 0
    if HP.RESUME_CHECKPOINT is not None and os.path.exists(HP.RESUME_CHECKPOINT):
        print(f"\n[RESUMING] Loading checkpoint from: {HP.RESUME_CHECKPOINT}")
        checkpoint = torch.load(HP.RESUME_CHECKPOINT, map_location=HP.DEVICE)
        start_iteration = load_checkpoint(model, optimizer, checkpoint,
                                          HP.DEVICE, ema_helper, deformation_sampler, scheduler)
        if HP.USE_AMP and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"✓ Resumed from iteration {start_iteration}")

    if HP.USE_WANDB:
        wandb.init(project=HP.WANDB_PROJECT, name=HP.EXPERIMENT_NAME, config=vars(HP))

    # ── Training loop ─────────────────────────────────────────────────────────
    global_iteration = start_iteration
    train_iter = iter(train_loader)
    best_models_psnr = deque(maxlen=3)
    best_models_ssim = deque(maxlen=3)
    pbar = tqdm(initial=start_iteration, total=HP.N_ITERS, desc="Training")

    while global_iteration < HP.N_ITERS:
        model.train()
        deformation_sampler.train()

        # ── Batch ─────────────────────────────────────────────────────────────
        try:
            source_patch, target_patch, mask_patch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            source_patch, target_patch, mask_patch = next(train_iter)

        source_patch = source_patch.to(HP.DEVICE)
        target_patch = target_patch.to(HP.DEVICE)
        mask_patch   = mask_patch  .to(HP.DEVICE)

        # ── Forward ───────────────────────────────────────────────────────────
        n_samples = HP.N_DEFORMATION_SAMPLES

        if n_samples == 0:
            if HP.USE_AMP:
                with torch.cuda.amp.autocast():
                    loss_fm = flow_matching_loss(model, source_patch, target_patch)
                    loss    = loss_fm
            else:
                loss_fm = flow_matching_loss(model, source_patch, target_patch)
                loss    = loss_fm

        else:
            # Multi-deformation: warp source, keep target as conditioning
            source_expanded = source_patch.repeat(n_samples, 1, 1, 1, 1)
            target_expanded = target_patch.repeat(n_samples, 1, 1, 1, 1)

            if HP.USE_AMP:
                with torch.cuda.amp.autocast():
                    deformation, mu, logvar = deformation_sampler(source_expanded, sample=True)
                    source_warped = apply_deformation_3d(source_expanded, deformation)
                    loss_fm = flow_matching_loss(model, source_warped, target_expanded)

                    B = source_patch.shape[0]
                    loss_kl = -0.5 * torch.mean(
                    1 + logvar - mu.pow(2) - logvar.exp())
                    dx = deformation[:, :, :, :, 1:] - deformation[:, :, :, :, :-1]
                    dy = deformation[:, :, :, 1:, :] - deformation[:, :, :, :-1, :]
                    dz = deformation[:, :, 1:, :, :] - deformation[:, :, :-1, :, :]
                    loss_smooth = torch.mean(dx**2) + torch.mean(dy**2) + torch.mean(dz**2)
                    loss = loss_fm + HP.KL_WEIGHT * loss_kl + HP.SMOOTH_WEIGHT * loss_smooth  #Fixed KL weight (posterior collapse may happen)
                    # kl_weight = HP.KL_WEIGHT * min(1.0, global_iteration / 50000)   # Gradual KL weight (Helps prevent posterior collapse)
                    # loss = loss_fm + kl_weight * loss_kl + HP.SMOOTH_WEIGHT * loss_smooth
            else:
                deformation, mu, logvar = deformation_sampler(source_expanded, sample=True)
                source_warped = apply_deformation_3d(source_expanded, deformation)
                loss_fm = flow_matching_loss(model, source_warped, target_expanded)

                B = source_patch.shape[0]
                loss_kl = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp()
            )
                dx = deformation[:, :, :, :, 1:] - deformation[:, :, :, :, :-1]
                dy = deformation[:, :, :, 1:, :] - deformation[:, :, :, :-1, :]
                dz = deformation[:, :, 1:, :, :] - deformation[:, :, :-1, :, :]
                loss_smooth = torch.mean(dx**2) + torch.mean(dy**2) + torch.mean(dz**2)
                loss = loss_fm + HP.KL_WEIGHT * loss_kl + HP.SMOOTH_WEIGHT * loss_smooth
                # kl_weight = HP.KL_WEIGHT * min(1.0, global_iteration / 50000)
                # loss      = loss_fm + kl_weight * loss_kl + HP.SMOOTH_WEIGHT * loss_smooth

        if HP.USE_AMP:
            scaler.scale(loss).backward()
            if (global_iteration + 1) % HP.GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        else:
            loss.backward()
            if (global_iteration + 1) % HP.GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        ema_helper.update(model)
        global_iteration += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}", def_scale=f"{deformation.abs().mean().item():.4f}")

        # ── WandB ─────────────────────────────────────────────────────────────
        if HP.USE_WANDB and global_iteration % 100 == 0:
            log_dict = {"train_loss": loss.item(), "loss_fm": loss_fm.item(),
                        "iteration": global_iteration}
            if n_samples > 0:
                log_dict.update({"loss_kl": loss_kl.item(), "loss_smooth": loss_smooth.item()})
            wandb.log(log_dict)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if global_iteration % HP.SNAPSHOT_FREQ == 0:
            ckpt_path = f"{base_path}/checkpoints/iter_{global_iteration:07d}.pth"
            save_checkpoint(model, optimizer, global_iteration, ckpt_path,
                            ema_helper, deformation_sampler, scaler, scheduler)
            pbar.write(f"✓ Checkpoint saved: iter_{global_iteration}")

        # ── Validation ────────────────────────────────────────────────────────
        if global_iteration % HP.VALIDATION_FREQ == 0:
            pbar.write(f"\n[Validation at iter {global_iteration}]")
            model.eval()
            deformation_sampler.eval()

            ema_helper.store(model.named_parameters())
            ema_helper.ema(model)

            val_metrics = {'PSNR': [], 'SSIM': [], 'MSE': [], 'MAE': []}

            with torch.no_grad():
                for val_batch_idx, (source_patch, target_patch, mask_patch) in enumerate(val_loader):
                    source_patch = source_patch.to(HP.DEVICE)
                    target_patch = target_patch.to(HP.DEVICE)
                    mask_patch = mask_patch  .to(HP.DEVICE)

                    if HP.N_DEFORMATION_VAL == 0:
                        target_fake = sample_flow_matching_3d(
                            model, source_patch,
                            steps=HP.FLOW_STEPS, method=HP.FLOW_METHOD, device=HP.DEVICE
                        )
                    else:
                        target_fake, _, _ = sample_flow_matching_ensemble_3d(
                            model, deformation_sampler, source_patch,
                            n_samples=HP.N_DEFORMATION_VAL,
                            steps=HP.FLOW_STEPS, method=HP.FLOW_METHOD, device=HP.DEVICE
                        )

                    mask_binary = (mask_patch > 0.5).float()
                    target_fake_masked = target_fake  * mask_binary
                    target_real_masked = target_patch * mask_binary

                    for b in range(target_fake.shape[0]):
                        metrics = compute_3d_metrics(
                            target_fake_masked[b, 0].cpu().numpy(),
                            target_real_masked[b, 0].cpu().numpy(),
                            mask_binary       [b, 0].cpu().numpy(),
                        )
                        val_metrics['PSNR'].append(metrics['psnr'])
                        val_metrics['SSIM'].append(metrics['ssim'])
                        val_metrics['MSE'] .append(metrics['mse'])
                        val_metrics['MAE'] .append(metrics['mae'])

                    if val_batch_idx < 10:
                        vis_path = f"{base_path}/validation/iter_{global_iteration:07d}_batch{val_batch_idx:03d}.png"
                        save_middle_slice_visualization(
                            source_patch, target_patch, target_fake,
                            mask_patch, vis_path, global_iteration
                        )
                        if HP.USE_WANDB:
                            wandb.log({"val_image": wandb.Image(vis_path),
                                       "iteration": global_iteration})

            avg_psnr = float(np.mean(val_metrics['PSNR']))
            avg_ssim = float(np.mean(val_metrics['SSIM']))
            avg_mse = float(np.mean(val_metrics['MSE']))
            avg_mae = float(np.mean(val_metrics['MAE']))
            pbar.write(f"  PSNR: {avg_psnr:.4f}  SSIM: {avg_ssim:.4f}  "
                       f"MSE: {avg_mse:.6f}  MAE: {avg_mae:.4f}")

            if HP.USE_WANDB:
                wandb.log({"val_psnr": avg_psnr, "val_ssim": avg_ssim,
                           "val_mse":  avg_mse,  "val_mae":  avg_mae,
                           "iteration": global_iteration})

            os.makedirs(f"{base_path}/validation", exist_ok=True)
            results = {'iteration': global_iteration,
                       'psnr': avg_psnr, 'ssim': avg_ssim,
                       'mse':  avg_mse,  'mae':  avg_mae,
                       'num_patches': len(val_metrics['PSNR'])}
            with open(f"{base_path}/validation/metrics_iter_{global_iteration:07d}.json", 'w') as f:
                json.dump(results, f, indent=2)

            best_models_psnr.append((avg_psnr, f"{base_path}/best/psnr_iter_{global_iteration:07d}.pth"))
            best_models_ssim.append((avg_ssim, f"{base_path}/best/ssim_iter_{global_iteration:07d}.pth"))
            best_models_psnr = deque(sorted(best_models_psnr, reverse=True)[:3], maxlen=3)
            best_models_ssim = deque(sorted(best_models_ssim, reverse=True)[:3], maxlen=3)

            if f"{base_path}/best/psnr_iter_{global_iteration:07d}.pth" in [m[1] for m in best_models_psnr]:
                save_checkpoint(model, optimizer, global_iteration,
                                f"{base_path}/best/psnr_iter_{global_iteration:07d}.pth",
                                ema_helper, deformation_sampler, scaler, scheduler)

            if f"{base_path}/best/ssim_iter_{global_iteration:07d}.pth" in [m[1] for m in best_models_ssim]:
                save_checkpoint(model, optimizer, global_iteration,
                                f"{base_path}/best/ssim_iter_{global_iteration:07d}.pth",
                                ema_helper, deformation_sampler, scaler, scheduler)

            ema_helper.restore(model.named_parameters())

    pbar.close()
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print(f"Total iterations: {HP.N_ITERS:,}")
    if best_models_psnr: print(f"Best PSNR: {best_models_psnr[0][0]:.4f}")
    if best_models_ssim: print(f"Best SSIM: {best_models_ssim[0][0]:.4f}")
    if HP.USE_WANDB: wandb.finish()


if __name__ == "__main__":
    main()