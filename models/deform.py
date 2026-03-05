# ============ 3D DEFORMATION NETWORK ============
class DeformationSampler3D(nn.Module):
    def __init__(self, input_channels=1, patch_size=96):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(32, 32, 3, stride=2, padding=1),  
            nn.SiLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),  
            nn.SiLU(),
            nn.Conv3d(64, 64, 3, stride=2, padding=1),  
            nn.SiLU(),
            nn.Conv3d(64, 128, 3, stride=2, padding=1), 
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1)  
        )
        self.fc_mu     = nn.Linear(128, 128)
        self.fc_logvar = nn.Linear(128, 128)
        self.fc_expand = nn.Linear(128, 128 * 2 * 2 * 2)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),  
            nn.SiLU(),
            nn.ConvTranspose3d(64, 64, 4, stride=2, padding=1),   
            nn.SiLU(),
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),   
            nn.SiLU(),
            nn.ConvTranspose3d(32, 32, 4, stride=2, padding=1),   
            nn.SiLU(),
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1),   
            nn.SiLU(),
            nn.ConvTranspose3d(16, 3, 3, stride=1, padding=1),    
        )
        self.final_upsample   = nn.Upsample(size=patch_size, mode='trilinear', align_corners=True)
        self.log_deform_scale = nn.Parameter(torch.tensor(0.05).log())
    
    def forward(self, x, sample=True):
        B        = x.size(0)
        features = self.encoder(x).view(B, 128)
        mu       = self.fc_mu(features)
        logvar   = self.fc_logvar(features)
        z        = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar) if sample else mu
        z        = self.fc_expand(z).view(B, 128, 2, 2, 2)
        deform   = self.decoder(z)
        deform   = self.final_upsample(deform)
        deform   = torch.tanh(deform) * self.log_deform_scale.exp()
        return deform, mu, logvar
        
def apply_deformation_3d(volume, deformation):
    B, C, D, H, W = volume.shape
    grid_d, grid_h, grid_w = torch.meshgrid(
        torch.linspace(-1, 1, D, device=volume.device),
        torch.linspace(-1, 1, H, device=volume.device),
        torch.linspace(-1, 1, W, device=volume.device),
        indexing='ij'
    )
    grid = torch.stack([grid_w, grid_h, grid_d], dim=0)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)
    
    deformation_reordered = torch.stack([
        deformation[:, 2, ...],
        deformation[:, 1, ...],
        deformation[:, 0, ...]
    ], dim=1)
    
    grid_warped = (grid + deformation_reordered).permute(0, 2, 3, 4, 1)
    return F.grid_sample(volume, grid_warped, mode='bilinear',
                         padding_mode='border', align_corners=True)