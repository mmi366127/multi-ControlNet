

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import argparse
import torch
import os


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model


def get_lr_scheduler(hparams):
    pass

def get_opts():

    parser = argparse.ArgumentParser()

    # ControlNet settings
    parser.add_argument("--only_mid_control", action="store_true",
                        help="only use mid control or not.")
    
    parser.add_argument("--sd_locked", type=bool, default=True,
                        help="Whether to lock the unet")
    # parser.add_argument("--sd_locked", action="store_false",
    #                     help="Whether to lock the unet.")

    # model config path
    parser.add_argument("--config", type=str,
                        default=None,
                        help="The config of the loaded model.")


    # directories
    parser.add_argument("--resume", type=str, 
                        default=None,
                        help="resume from the given path")
    
    parser.add_argument("--root_dir", type=str,
                        default="/home/lolicon/data/experiments/illya-takina-dreambooth",
                        help="root directory of experiment")


    # training arguments
    parser.add_argument('--lr', type=float, default=1e-6,
                        help='learning rate')

    parser.add_argument("--scale_lr", action="store_true",
                        help="scale base-lr by ngpu * batch_size * n_accumulate")

    parser.add_argument("--precision", type=str, default="16",
                        help="The precision of training.")

    parser.add_argument("--batch_size", type=int, default=None,
                        help="batch size")
    
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    # resume weights
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='pretrained checkpoint path to load')
    
    # Stable-Diffusion weight
    parser.add_argument('--sd_path', type=str, default='/home/lolicon/workspace/stable-diffusion-webui/models/Stable-diffusion/CounterfeitV30_v30.safetensors',
                        help='the pretrained Stable-Diffusion weight path')
    

    # other arguments
    parser.add_argument("--seed", type=int, default=9487,
                        help="the seed for seed_everything")


    # not implemented yet 
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################

    

    return parser.parse_args()


