
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from pathlib import Path
import torch, einops
import random, cv2
import numpy as np
import json

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.dir =  Path(data_path)
        self.data = []
        with open(self.dir / Path('info.json'), 'rt')  as f:
            for line in f:
                json_obj = json.loads(line)
                self.data.append(json_obj)
        
        config = OmegaConf.load(self.dir / 'model.yaml')
        
        self.image_keys = [config.model.params.first_stage_key]
        for key, value in config.model.params.control_stage_config.params.image_control_keys.items():
            self.image_keys.append(value)
        
        self.text_keys = [config.model.params.cond_stage_key]
        for key, value in config.model.params.control_stage_config.params.text_control_keys.items():
            self.text_keys.append(value)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        info = self.data[idx]

        # random flip
        to_flip = random.random() > 0.5

        ret = {}
        for key, value in info.items():
            if key in self.image_keys:
                # image path
                source = cv2.imread(str(self.dir / value))
                source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
                source = cv2.resize(source, (512, 512))
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
    