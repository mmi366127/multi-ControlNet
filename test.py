from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config
from cldm.model import load_state_dict
from omegaconf import OmegaConf
import pytorch_lightning as pl
from packaging import version
from utils import get_opts
import argparse
import datetime
import sys, os
import torch



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
    
        if main_ckpt_path is not None:
            sd_ckpt = load_state_dict(main_ckpt_path)
        else:
            sd_ckpt = {}

        sd_control = {}
        for model_name, cfg in config.model.params.control_stage_config.params.control_nets.items():
            if "weight" in cfg.keys():
                sd_ctrl = load_state_dict(cfg["weight"])
                sd_control[model_name] = sd_ctrl
        
        model.load_multi_state_dict(sd_ckpt, **sd_control)

    return model

    
def main(opts, now):
    total = 0
    sd = load_state_dict('/home/lolicon/data/loraDreambooth-new/exp2023-05-2022-52-00/checkpoints/epoch=000007.ckpt')
    for key, value in sd.items():
        if 'up' in key :
            total += torch.sum(value).detach().cpu().numpy()
    print(total)

    exit(0)
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


    # prepare model
    model = prepare_model(opts, config)


    print(model.configure_optimizers())
    
   


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
