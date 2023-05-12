# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py


import pytorch_lightning as pl
from typing import List
import os, math, torch


class LoRAModule(pl.LightningModule):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(self, lora_name, org_module: torch.nn.Module, multiplier=1.0, lora_dim=4, alpha=1):
        """ if alpha == 0 or None, alpha is rank (no scaling). """
        super().__init__()
        
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        
        # determine replace module type
        if org_module.__class__.__name__ == 'Conv2d':
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            self.lora_down = torch.nn.Conv2d(in_dim, lora_dim, (1, 1), bias=False)
            self.lora_up = torch.nn.Conv2d(lora_dim, out_dim, (1, 1), bias=False)
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = torch.nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(lora_dim, out_dim, bias=False)

        # determine scale
        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer('alpha', torch.tensor(alpha))
        
        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a = math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)
        
        self.multiplier = multiplier
        self.org_module = org_module
        
    
    def unload(self):
        # unload lora module for original module
        self.org_module.forward = self.org_forward
        
    
    def apply_to(self):
        # apply lora to original module
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.org_forward
    
    
    def forward(self, x):
        return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale    
    
      
class LoRANetwork(pl.LightningModule):
    UNET_TARGET_REPLACE_MODULE = ["SpatialTransformer"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER = 'lora_te'

    def __init__(self, text_encoder=None, unet=None, multiplier=1.0, lora_dim=4, alpha=1, weights_sd=None) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha

        # create module instances
        def create_modules(prefix, root_module: torch.nn.Module, target_replace_modules) -> List[LoRAModule]:
            loras = []
            for name, module in root_module.named_modules():
                    if module.__class__.__name__ in target_replace_modules:
                        for child_name, child_module in module.named_modules():
                            if child_module.__class__.__name__ == "Linear" or (child_module.__class__.__name__ == "Conv2d" and child_module.kernel_size == (1, 1)):
                                lora_name = prefix + '.' + name + '.' + child_name
                                lora_name = lora_name.replace('.', '_')
                                lora = LoRAModule(lora_name, child_module, self.multiplier, self.lora_dim, self.alpha)
                                loras.append(lora)
            return loras

        names = set()

        if text_encoder is not None:
            self.text_encoder_loras = create_modules(LoRANetwork.LORA_PREFIX_TEXT_ENCODER, text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)            
            print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")
        else:
            self.text_encoder_loras = []


        if unet is not None:
            self.unet_loras = create_modules(LoRANetwork.LORA_PREFIX_UNET, unet, LoRANetwork.UNET_TARGET_REPLACE_MODULE)
            print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")
        else:
            self.unet_loras = []

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)
            
        self.weights_sd = weights_sd
    
    def set_mulitiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier    
    
    def load_weights(self, file):
        if os.path.splitext(file)[1] == '.safetensors':
            from safetensors.torch import load_file, safe_open
            self.weights_sd = load_file(file)
        else:
            self.weights_sd = torch.load(file, map_location='cpu')
            
    def apply_to(self, apply_text_encoder=None, apply_unet=None):
        if self.weights_sd:
            weights_has_text_encoder = weights_has_unet = False
            for key in self.weights_sd.keys():
                if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                    weights_has_text_encoder = True
                elif key.startswith(LoRANetwork.LORA_PREFIX_UNET):
                    weights_has_unet = True

            if apply_text_encoder is None:
                apply_text_encoder = weights_has_text_encoder
            else:
                assert apply_text_encoder == weights_has_text_encoder, f"text encoder weights: {weights_has_text_encoder} but text encoder flag: {apply_text_encoder}"

            if apply_unet is None:
                apply_unet = weights_has_unet
            else:
                assert apply_unet == weights_has_unet, f"u-net weights: {weights_has_unet} but u-net flag: {apply_unet}"
        else:
            assert apply_text_encoder is not None and apply_unet is not None, f"internal error: flag not set"

        if apply_text_encoder:
            print("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        if self.weights_sd:
            # if some weights are not in state dict, it is ok because initial LoRA does nothing (lora_up is initialized by zeros)
            info = self.load_state_dict(self.weights_sd, False)
            print(f"weights are loaded: {info}")

    def enable_gradient_checkpointing(self):
        # not supported
        pass

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr):
        def enumerate_params(loras):
            params = []
            for lora in loras:
                params.extend(lora.parameters())
            return params

        self.requires_grad_(True)
        all_params = []

        if self.text_encoder_loras:
            param_data = {'params': enumerate_params(self.text_encoder_loras)}
            if text_encoder_lr is not None:
                param_data['lr'] = text_encoder_lr
            all_params.append(param_data)

        if self.unet_loras:
            param_data = {'params': enumerate_params(self.unet_loras)}
            if unet_lr is not None:
                param_data['lr'] = unet_lr
            all_params.append(param_data)

        return all_params

    def prepare_grad_etc(self):
        self.requires_grad_(True)

    def on_train_epoch_start(self):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == '.safetensors':
            from safetensors.torch import save_file

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def load_module(self):
        for lora in self.unet_loras + self.text_encoder_loras:
            lora.apply_to()

    def unload_module(self):
        for lora in self.unet_loras + self.text_encoder_loras:
            lora.unload()




