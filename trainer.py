
from pytorch_lightning.callbacks import ModelCheckpoint
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from torch.utils.data import DataLoader
from data.dataset import CustomDataset
from cldm.logger import ImageLogger
import pytorch_lightning as pl
from pathlib import Path
import argparse
import torch


class myLogger(ImageLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")


def main(hparams):

    root_dir = Path(hparams.root_dir)

    model = create_model(str(root_dir / 'model.yaml')).cpu()
    
    model.sd_locked = hparams.sd_locked
    model.only_mid_control = hparams.only_mid_control
    model.learning_rate = hparams.lr

    # weight for unet, encoder, decoder, text embedding 
    main_ckpt_path = hparams.unet_path
    main_ckpt_path = '../stable-diffusion-webui/models/Stable-diffusion/pastelMixStylizedAnime_pastelMixFull.safetensors'

    # weight for control net 
    ctrl_pose_path = hparams.ctrl_path
    ctrl_pose_path = '../stable-diffusion-webui/models/ControlNet/control_sd15_openpose.pth'

    # pretrained weight
    sd_ckpt = load_state_dict(main_ckpt_path)
    sd_ctrl = load_state_dict(ctrl_pose_path)
    model.load_multi_state_dict(sd_ckpt, pose_model=sd_ctrl)


    # dataset
    dataset = CustomDataset(root_dir)
    dataloader = DataLoader(dataset, num_workers=8, batch_size=hparams.batch_size, shuffle=True)


    torch.set_float32_matmul_precision('medium')

    # callbacks
    ckpt_cb = ModelCheckpoint(dirpath=Path(root_dir) / 'ckpt',
                              filename='{epoch:d}',
                              monitor='train/loss',
                              mode='min',
                              every_n_epochs=100,
                              save_top_k=3)
    
    logger = myLogger(batch_frequency=hparams.logger_freq)
    callbacks = [ckpt_cb, logger]
    
    trainer = pl.Trainer(max_epochs=hparams.num_epochs,
                         callbacks=callbacks,
                         log_every_n_steps=20,
                         resume_from_checkpoint=hparams.ckpt_path,
                         precision=hparams.precision,
                         enable_model_summary=False,
                         accelerator="auto",
                         devices=hparams.num_gpus,
                         num_sanity_val_steps=1,
                         benchmark=True,
                         profiler="simple" if hparams.num_gpus==1 else None,
                         enable_checkpointing=True)

    trainer.fit(model, dataloader)


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_mid_control", action="store_true",
                        help="only use mid control or not.")
    
    parser.add_argument("--sd_locked", action="store_false",
                        help="Whether to lock the unet.")

    parser.add_argument("--root_dir", type=str,
                        default="/home/lolicon/data/dataset/lycoris",
                        help="root directory of dataset.")

    parser.add_argument("--precision", type=str, default="16",
                        help="The precision of training.")

    parser.add_argument("--batch_size", type=int, default=2,
                        help="batch size")

    parser.add_argument("--num_epochs", type=int, default=2000,
                        help="number of training epochs")
    
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')
    
    parser.add_argument('--unet_path', type=str, default=None,
                        help='the pretrained unet weight path')

    parser.add_argument('--ctrl_path', type=str, default=None,
                        help='the pretrained ControlNet weight path')
    
    parser.add_argument("--logger_freq", type=int, default=2000,
                        help="the frequency to log image.")

    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')

    # not used yet 
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

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    return parser.parse_args()


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)

