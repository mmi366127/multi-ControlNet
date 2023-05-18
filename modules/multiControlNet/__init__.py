
from ldm.modules.diffusionmodules.util import timestep_embedding
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config, log_txt_as_img, exists
from ldm.models.diffusion.ddim import DDIMSampler

from torchvision.utils import make_grid
import torch.nn as nn
import torch

from typing import Union
from omegaconf import OmegaConf
from einops import rearrange, repeat
from pathlib import Path
import einops, os


class MultiControlNet(nn.Module):
    def __init__(
            self, 
            control_nets,
            cond_keys,
            image_control_keys,
            text_control_keys
    ):
        """
            Wrap control nets so that it can support multiple nets and prompts
            Args:
                control_nets(dict{str: str}) 
                    the name of the control net and the config
                cond_keys(dict{str: str})
                    the name of the control net input and the corresponding model
        """
        super().__init__()
        self.control_nets = []
        self.cond_keys = cond_keys
        self.image_control_keys = image_control_keys
        self.text_control_keys = text_control_keys

        for model_name in control_nets.keys():
            model = instantiate_from_config(control_nets[model_name])
            self.control_nets.append(model_name)
            setattr(self, model_name, model)

        # control scale not added to config yet
        self.control_scales = {
            name: [1.0] * 13 for name in self.cond_keys.keys()
        } 

    def forward(self, x, hint, timesteps, context, **kwargs):
        """
            forward mutiplt context and hint for multiple control net
            Args:
                x(torch.Tensor)
                    the noise to predict
                hint(dict{str: torch.Tensor})
                    the image condition 
                timesteps(torch.Tensor, float)
                    the current diffusion timestep
                context(torch.Tensor)
                    the main text condition of the UNET, used when there's no corresponding condition
        """
        control = None
        for cond_key, model_name in self.cond_keys.items():
            if cond_key in hint.keys():
                hint__ = hint[self.image_control_keys[cond_key]]
                context__ = hint[self.text_control_keys[cond_key]] if self.text_control_keys[cond_key] in hint.keys() else [context]
                if control is None:    
                    control = getattr(self, model_name)(x, torch.cat(hint__, 1), timesteps, torch.cat(context__, 1))
                    control = [c * scale for c, scale in zip(control, self.control_scales[cond_key])]
                else:
                    control_ = getattr(self, model_name)(x, torch.cat(hint__, 1), timesteps, torch.cat(context__, 1))
                    control = [
                        prev_control + scale * curr_control 
                        for prev_control, curr_control, scale in zip(control, control_, self.control_scales[cond_key])
                    ]
    
        return control

class MultiControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, only_mid_control, *args, **kwargs):
        r"""
            A multi-ControlNet version of ControlLDM
            control_stage_configs
                config that used to instantiate the multi-control model
            only_mid_control
                use only mid block or not
        """
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.only_mid_control = only_mid_control
        

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        ret = {'c_crossattn': [c]}
        for control_key in batch.keys():
            if control_key in self.control_model.image_control_keys.values():
                
                control = batch[control_key]
                if bs is not None:
                    control = control[:bs]
                control = control.to(self.device)
                control = einops.rearrange(control, 'b h w c -> b c h w')
                control = control.to(memory_format=torch.contiguous_format).float()
                ret[control_key] = [control]

            elif control_key in self.control_model.text_control_keys.values():

                control = batch[control_key]
                control = self.get_learned_conditioning(control)

                ret[control_key] = [control]

        return x, ret

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        
        # multi-control
        control = self.control_model(x=x_noisy, hint=cond, timesteps=t, context=cond_txt)
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=25, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=7.0, unconditional_guidance_label=None,
                   use_ema_scope=True, size=512,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["conditioning"] = log_txt_as_img((size, size), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond=c,
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_full = c
            uc_full["c_crossattn"] = [uc_cross]
            samples_cfg, _ = self.sample_log(cond=c,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full, size=size
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, size=512, **kwargs):
        ddim_sampler = DDIMSampler(self)
        shape = (self.channels, size // 8, size // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    def load_multi_state_dict(self, diffusion_state_dict, **kwargs):
        
        state_dict = {}

        control_state_dicts = kwargs

        for key, weight in diffusion_state_dict.items():
            if key.startswith("control_model."): continue
            state_dict[key] = weight
        
        for model_name, control_state_dict in control_state_dicts.items():
            for key, weight in control_state_dict.items():
                if key.startswith("control_model."):
                    new_key = ("." + model_name + ".").join(key.split('.', 1))
                    state_dict[new_key] = weight
        
        self.load_state_dict(state_dict, strict=False)

    def multi_state_dict(self):

        state_dict = self.state_dict()

        ret = {key: {} for key in self.control_model.control_nets + ['main']}
        
        for key, weight in state_dict.items():
            if key.startswith("control_model."):
                temp = key.split('.', 2)
                model_name = temp[1]
                new_key = ".".join([temp[0], temp[2]])
                ret[model_name][new_key] = weight
            else:
                ret['main'][key] = weight
        
        return ret

    def save_weights(self, save_dir, only_control=True):
        
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        state_dicts = self.multi_state_dict()
        os.makedirs(save_dir, exist_ok=True)
        
        def save_(state_dict, path):
            to_safetensors = str(path).rsplit('.', 1)[-1] == 'safetensors'
            if to_safetensors:
                from safetensors.torch import save_file
                save_file(state_dict, path)
            else:
                torch.save(state_dict, path)
        

        for key, state_dict in state_dicts.items():
            if key == 'main':
                if not only_control:
                    save_(state_dict, save_dir / 'diffusion_model.safetensors')
            else:
                save_(state_dict, save_dir / '.'.join([key, 'safetensors']))
    
    def on_train_batch_start(self, batch, batch_idx):
        super().on_train_batch_start(batch, batch_idx, None)

