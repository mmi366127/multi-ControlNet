
from utils.callbacks import CUDACallback, ImageLogger, SetupCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from torch.utils.data import DataLoader
from data.dataset import CustomDataset
from packaging import version
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
from pathlib import Path
import argparse
import torch
import sys, os
import datetime
from omegaconf import OmegaConf
from utils import get_opts

from ldm.util import log_txt_as_img, exists, instantiate_from_config
from modules.multiControlNet import MultiControlLDM
    
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
                   use_ema_scope=True, size=512,
                   **kwargs):   
        batch = batch[0]
        return super().log_images(batch, N=N, n_row=n_row, sample=sample, ddim_step=ddim_steps, ddim_eta=ddim_eta, 
                    return_keys=return_keys, quantize_denoised=quantize_denoised, inpaint=inpaint, 
                    plot_denoise_rows=plot_denoise_rows, plot_progressive_rows=plot_progressive_rows,
                    use_ema_scope=use_ema_scope, size=size, **kwargs)


def prepare_model(opts, config):

    # load model
    model = instantiate_from_config(config["model"]).cpu()

    # basic settings
    model.sd_locked = opts.sd_locked
    model.only_mid_control = opts.only_mid_control
    model.learning_rate = opts.lr

    # load weight
    if opts.resume_from_checkpoint:    
        sd = load_state_dict(opts.resume_from_checkpoint)
        model.load_state_dict(sd)

    else:
        # weight for unet, encoder, decoder, text embedding 
        main_ckpt_path = opts.sd_path
        main_ckpt_path = '../stable-diffusion-webui/models/Stable-diffusion/pastelMixStylizedAnime_pastelMixFull.safetensors'
    
        if main_ckpt_path is not None:
            sd_ckpt = load_state_dict(main_ckpt_path)

        sd_control = {}
        for model_name, cfg in config.model.params.control_stage_config.params.control_nets.items():
            if "weight" in cfg.keys():
                sd_ctrl = load_state_dict(cfg["weight"])
                sd_control[model_name] = sd_ctrl
        
        model.load_multi_state_dict(sd_ckpt, **sd_control)

    return model

def prepare_callbacks(opts, config, lightning_config, model, now):

    trainer_kwargs = {}

    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "name": opts.exp_name,
                "save_dir": opts.root_dir,
                "id": opts.root_dir.split("/")[-1],
            }
        },
        "TensorBoardLogger": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "save_dir": opts.root_dir,
            }
        },
    }    
    # use TensorBoardLogger as default logger
    default_logger_cfg = default_logger_cfgs["TensorBoardLogger"]

    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)

    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
    

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": os.path.join(opts.root_dir, "checkpoints"),
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
        }
    }
    if "modelcheckpoint" in lightning_config:
        modelcheckpoint_cfg = lightning_config.modelcheckpoint
    else:
        modelcheckpoint_cfg = OmegaConf.create()

    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 3

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg =  OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")

    if version.parse(pl.__version__) < version.parse('1.4.0'):
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "utils.callbacks.SetupCallback",
            "params": {
                "resume": opts.resume,
                "now": now,
                "logdir": opts.root_dir,
                "ckptdir": os.path.join(opts.root_dir, "checkpoints"),
                "cfgdir": os.path.join(opts.root_dir, "configs"),
                "config": config,
                "lightning_config": lightning_config
            }
        },
        "image_logger": {
            "target": "utils.callbacks.ImageLogger",
            "params": {
                "batch_frequency": 750,
                "max_images": 4,
                "clamp": True
            }
        },
        "cuda_callback": {
            "target": "utils.callbacks.CUDACallback"
        }
    }
    if version.parse(pl.__version__) >= version.parse('1.4.0'):
        default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
        print('Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint': {                              
                "target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                'params': {
                        "dirpath": os.path.join(opts.root_dir, 'checkpoints', 'trainstep_checkpoints'),
                        "filename": "{epoch:06}-{step:09}",
                        "verbose": True,
                        'save_top_k': -1,
                        'every_n_train_steps': 10000,
                        'save_weights_only': True
                }
            }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    # trainer config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config
    
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
        callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
    elif 'ignore_keys_callback' in callbacks_cfg:
        del callbacks_cfg['ignore_keys_callback']

    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    return trainer_kwargs, trainer_opt
    
def prepare_dataset(config):
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    return data
    
def main(opts, now):

    
    # setup seed
    seed_everything(opts.seed)

    # torch settings
    torch.set_float32_matmul_precision('medium')
    
    # read config
    config_path = opts.config
    config = OmegaConf.load(config_path)
    lightning_config = config.pop("lightning", OmegaConf.create())
    
    # configured learning rate, batch_size and lr in opts can overwrite those in config
    if opts.batch_size is not None: 
        config.data.params.batch_size = opts.batch_size
    if opts.lr is not None:
        config.model.base_learning_rate = opts.lr


    # prepare model
    model = prepare_model(opts, config)
    
    # prepare callbacks
    trainer_kwargs, trainer_opt = prepare_callbacks(opts, config, lightning_config, model, now)

    # update trainer_kwargs
    trainer_kwargs["log_every_n_steps"] = 20
    trainer_kwargs["precision"] = opts.precision
    trainer_kwargs["accelerator"] = "auto"
    trainer_kwargs["devices"] = opts.num_gpus
    trainer_kwargs["num_sanity_val_steps"] = 1
    trainer_kwargs["benchmark"] = True
    trainer_kwargs["profiler"] = "simple" if opts.num_gpus == 1 else None
    trainer_kwargs["enable_checkpointing"] = True
        

    trainer = pl.Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = opts.root_dir 

    
    # gradient accumulation
    ngpu = opts.num_gpus
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opts.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")
                        

    # dataset
    dataModule = prepare_dataset(config)
    trainer.fit(model, dataModule)


if __name__ == '__main__':

    # time stamp
    now = datetime.datetime.now().strftime("%Y-%m-%d%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    sys.path.append(os.getcwd())
    
    opts = get_opts()

    if opts.exp_name and opts.resume:
        raise ValueError(
            "-n/--exp_name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    
    if opts.resume:
        if not os.path.exists(opts.resume):
            raise ValueError("Cannot find {}".format(opts.resume))
        if os.path.isfile(opts.resume):
            paths = opts.resume.split("/")
            logdir = "/".join(paths[:-2])
            ckpt = opts.resume
        else:
            assert os.path.isdir(opts.resume), opts.resume
            logdir = opts.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opts.root_dir = logdir
        opts.resume_from_checkpoint = ckpt
    
    else:
        nowname = opts.exp_name + now
        logdir = os.path.join(opts.root_dir, nowname)
        opts.root_dir = logdir
    
    main(opts, now)
