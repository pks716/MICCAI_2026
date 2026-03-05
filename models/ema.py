import torch.nn as nn

"""
A method that increases the stability of a model's convergence and helps it reach a better overall solution by preventing convergence to a local minima. 
To avoid drastic changes in the model's weights during training, a copy of the current weights is created before updating the model's weights. 
Then the model's weights are updated to be the weighted average between the current weights and the post-optimization step weights.
"""
class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}
        self.backup = {} 
    
    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1. - self.mu) * param.data + self.mu * self.shadow[name].data
    
    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)
    
    # Store current model weights before applying EMA
    def store(self, parameters):
        """
        Save the current parameters for restoration later.
        Args:
            parameters: model.parameters()
        """
        self.backup = {}
        for name, param in parameters:
            if param.requires_grad:
                self.backup[name] = param.data.clone()
    
    # Restore original model weights after validation
    def restore(self, parameters):
        """
        Restore the parameters that were saved with store().
        Args:
            parameters: model.parameters()
        """
        for name, param in parameters:
            if param.requires_grad:
                if name in self.backup:
                    param.data.copy_(self.backup[name])
        self.backup = {}
    
    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(
                inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        # module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy
    
    def state_dict(self):
        return self.shadow
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict