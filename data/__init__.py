from torch.utils.data import Dataset, DataLoader
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import pytorch_lightning as pl
from functools import partial
from pathlib import Path
import torch, einops
import random, cv2
import numpy as np
import json

training_templates = [
    'an illustration of {}',
    'an illustration of a cute {}',
    'a depiction of a cute {}'
]

reg_templates = [
    'an illustration of {}',
    'an illustration of the cute {}',
    'a depiction of the cute {}'
]

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, reg=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        
        # is dreambooth or not
        self.dreambooth = False
        if train is not None and reg is not None:
            self.dreambooth = True

        if train is not None:
            self.dataset_configs["train"] = train
        if reg is not None:
            self.dataset_configs["reg"] = reg
        
        self.train_dataloader = self._train_dataloader
        
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        if self.dreambooth:
            train_set = self.datasets["train"]
            reg_set = self.datasets["reg"]
            concat_dataset = ConcatDataset(train_set, reg_set)
            return DataLoader(concat_dataset, batch_size=self.batch_size,
                            num_workers=self.num_workers, shuffle=True)
        else:
            train_set = self.datasets["train"]
            return DataLoader(concat_dataset, batch_size=self.batch_size,
                            num_workers=self.num_workers, shuffle=True)

    def _val_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=shuffle)


class CustomDataset(Dataset):
    def __init__(self, 
                data_path, 
                size=512, 
                repeats=100,
                interpolation="bicubic",
                flip_p=0.5,
                set="train",
                center_crop=False):
        
        self.dir =  Path(data_path)
        self.data = []

        with open(self.dir / Path('info.json'), 'rt')  as f:
            for line in f:
                json_obj = json.loads(line)
                self.data.append(json_obj)
            
        self.num_images = len(self.data)
        self.length = self.num_images
        
        self.center_crop = center_crop
        self.interpolation = {
            "bilinear": cv2.INTER_LINEAR,
            "nearest": cv2.INTER_NEAREST,
            "area": cv2.INTER_AREA,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4
        }[interpolation]

        self.flip_p = flip_p
        self.size = size

        if set == "train":
            self.length = self.num_images * repeats
        

        # config = OmegaConf.load(self.dir / 'model.yaml')
        
        self.image_keys = ["jpg", "pose_1"]
        # for key, value in config.model.params.control_stage_config.params.image_control_keys.items():
        #     self.image_keys.append(value)
        
        self.text_keys = ["txt", "pose_1_text"]
        # for key, value in config.model.params.control_stage_config.params.text_control_keys.items():
        #     self.text_keys.append(value)
        
    def __len__(self):
        return int(self.length)

    def __getitem__(self, idx):

        info = self.data[idx % self.num_images]

        # random flip
        to_flip = random.random() < self.flip_p

        ret = {}
        for key, value in info.items():
            if key in self.image_keys:
                # image path
                source = cv2.imread(str(self.dir / value))
                source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
                
                # center crop
                if self.center_crop:
                    H, W, C = source.shape
                    crop = min(H, W)
                    source = source[(H - crop) // 2: (H + crop) // 2, 
                              (W - crop) // 2: (W + crop) // 2]

                # resize 
                source = cv2.resize(source, (self.size, self.size), interpolation=self.interpolation)

                # different preprocess for different control key
                if key == self.image_keys[0]: # source            
                    source = source.astype(np.float32) / 255.0
                else:
                    source = (source.astype(np.float32) / 127.5) - 1.0
                
                # random filp
                if to_flip:
                    source = cv2.flip(source, 1)

                ret[key] = source

            else:
                # text info
                text = value
                if key != self.text_keys[0]:
                    text = ', '.join([info[self.text_keys[0]], value])
                ret[key] = text
        
        return ret


class DreamBoothDataset(CustomDataset):
    def __init__(self, class_keyword="girl",
                 reg=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg = reg
        self.class_keyword = class_keyword
    
    def __getitem__(self, idx):

        info = self.data[idx % self.num_images]

        # random flip
        to_flip = random.random() < self.flip_p

        ret = {}
        for key, value in info.items():
            if key in self.image_keys:
                # image path
                source = cv2.imread(str(self.dir / value))
                source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
                
                # center crop
                if self.center_crop:
                    H, W, C = source.shape
                    crop = min(H, W)
                    source = source[(H - crop) // 2: (H + crop) // 2, 
                              (W - crop) // 2: (W + crop) // 2]

                # resize 
                source = cv2.resize(source, (self.size, self.size), interpolation=self.interpolation)

                # different preprocess for different control key
                if key == self.image_keys[0]: # source            
                    source = source.astype(np.float32) / 255.0
                else:
                    source = (source.astype(np.float32) / 127.5) - 1.0
                
                # random filp
                if to_flip:
                    source = cv2.flip(source, 1)

                ret[key] = source

            else:
                # text info    
                text = value
                if key != self.text_keys[0]:
                    if not self.reg:    
                        keyword = random.choice(training_templates).format(value)
                    else:
                        keyword = random.choice(reg_templates).format(self.class_keyword)
                    text = ', '.join([keyword, info[self.text_keys[0]]])
                ret[key] = text
        
        return ret
    

class MergeDataset(Dataset):
    def __init__(self, **kwargs):
        
        self.bin = []
        self.length = 0
        self.datasets = []
        for key, config in kwargs.items():
            dataset = instantiate_from_config(config)
            self.datasets.append(dataset)
            self.length += len(dataset)
            self.bin.append(self.length)

        self.bin = np.array(self.bin)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        dataset_idx = np.digitize(idx, self.bin)
        return self.datasets[dataset_idx][idx - self.bin[dataset_idx]]


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

