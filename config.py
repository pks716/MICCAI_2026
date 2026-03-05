# ============ CONFIGURATION ============

# For Smaller Capacity Model
class Config:
    TRAIN_PATCHES_DIR = None #Train_dir
    VAL_PATCHES_DIR   = None #Val_dir
    EXPERIMENT_NAME   = None #Exp_name
    RESUME_CHECKPOINT = None
    
    # Task: 'MR2CT' or 'CT2MR'
    # MR2CT dataloader returns: (mr_patch, ct_patch, mask)
    # CT2MR dataloader returns: (ct_patch, mr_patch, mask)
    TASK = 'MR2CT'

    # Training
    DEVICE        = 'cuda'
    BATCH_SIZE    = 4
    LEARNING_RATE = 2e-4
    N_ITERS       = 1000000

    GRAD_ACCUM_STEPS = 8
    
    RANDOM_SEED     = 42 
    VALIDATION_FREQ = 50000 
    SNAPSHOT_FREQ   = 50000 
    
    # 3D Patch
    PATCH_SIZE = 96
    
    # Multi-deformation (set 0 to disable)
    N_DEFORMATION_SAMPLES = 0
    N_DEFORMATION_VAL     = 0
    
    # Flow matching
    FLOW_STEPS  = 5
    FLOW_METHOD = 'euler'
    
    KL_WEIGHT     = 0.001
    SMOOTH_WEIGHT = 0.01
    
    # Mixed precision
    USE_AMP = True
    
    # WandB
    USE_WANDB     = False
    WANDB_PROJECT = "3D_Flow_Matching"
    
#For Bigger Capacity Model
class FlowMatchingConfig3D:
    def __init__(self):
        self.model = type('ModelConfig', (), {})()
        self.model.type = "sg"
        self.model.in_channels = 2   
        self.model.out_ch = 1      
        self.model.ch = 128
        self.model.ch_mult = [1, 2, 4, 4]
        self.model.num_res_blocks = 2
        self.model.attn_resolutions = [12]
        self.model.dropout = 0.1
        self.model.ema_rate = 0.9999
        self.model.ema = True
        self.model.resamp_with_conv = True
        
        self.data = type('DataConfig', (), {})()
        self.data.image_size = HP.PATCH_SIZE
        self.data.channels = 1
        
        self.diffusion = type('DiffusionConfig', (), {})()
        self.diffusion.num_diffusion_timesteps = 1000