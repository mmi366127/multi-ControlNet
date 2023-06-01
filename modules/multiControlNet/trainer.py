from modules.multiControlNet import MultiControlLDM
from lightning.pytorch.utilities import grad_norm
from ldm.util import instantiate_from_config
from modules.Lora import LoRA
import torch


class LoraEnsemble(MultiControlLDM):
    def __init__(self, use_unet=True, use_text_encoder=False, lora_multiplier=1.0, lora_dim=64, lora_alpha=4.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        unet = self.control_model if use_unet else None
        text_encoder = self.cond_stage_model if use_text_encoder else None
        self.lora = LoRA(unet=unet, text_encoder=text_encoder, multiplier=lora_multiplier, lora_dim=lora_dim, alpha=lora_alpha)

    def state_dict(self):
        # return state_dict to save
        ret = {}
        lora_sd = self.lora.state_dict()
        for key, value in lora_sd.items():
            if not 'org_module' in key:
                ret[key] = value
        return ret

    def configure_optimizers(self):
        lr = self.learning_rate
        params = self.lora.get_trainable_params()
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["loss"])

    # for dubug
    def on_before_optimizer_step(self, optimizer, _):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        lora_grad = grad_norm(self.lora, norm_type=2)
        self.log_dict(lora_grad)

class DreamBooth(MultiControlLDM):
    def __init__(self, reg_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_weight = reg_weight
        
    def training_step(self, batch, batch_idx):
        
        train_batch = batch[0]
        reg_batch = batch[1]
        
        loss_train, loss_dict = self.shared_step(train_batch)
        loss_reg, _ = self.shared_step(reg_batch)
        
        loss = loss_train + self.reg_weight * loss_reg

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss
    
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=25, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=7.0, unconditional_guidance_label=None,
                   use_ema_scope=True, size=256,
                   **kwargs):   
        batch = batch[0]
        return super().log_images(batch, N=N, n_row=n_row, sample=sample, ddim_step=ddim_steps, ddim_eta=ddim_eta, 
                    return_keys=return_keys, quantize_denoised=quantize_denoised, inpaint=inpaint, 
                    plot_denoise_rows=plot_denoise_rows, plot_progressive_rows=plot_progressive_rows,
                    use_ema_scope=use_ema_scope, size=size, **kwargs)


class LoraDreamBooth(LoraEnsemble):
    def __init__(self, reg_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_weight = reg_weight

    def training_step(self, batch, batch_idx):
        
        train_batch = batch[0]
        reg_batch = batch[1]
        
        loss_train, loss_dict = self.shared_step(train_batch)
        loss_reg, _ = self.shared_step(reg_batch)
        
        loss = loss_train + self.reg_weight * loss_reg

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=25, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=7.0, unconditional_guidance_label=None,
                   use_ema_scope=True, size=512,
                   **kwargs):   
        batch = batch[0]
        return super().log_images(batch, N=N, n_row=n_row, sample=sample, ddim_step=ddim_steps, ddim_eta=ddim_eta, 
                    return_keys=return_keys, quantize_denoised=quantize_denoised, inpaint=inpaint, 
                    plot_denoise_rows=plot_denoise_rows, plot_progressive_rows=plot_progressive_rows,
                    use_ema_scope=use_ema_scope, size=size, **kwargs)
    


