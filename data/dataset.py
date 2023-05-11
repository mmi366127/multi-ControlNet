

from torch.utils.data import Dataset

from annotator.util import HWC3
from annotator.canny import CannyDetector

from omegaconf import OmegaConf
from pathlib import Path
import numpy as np
import torch, einops
import json
import random
import cv2


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / max(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


apply_canny = CannyDetector()


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
        
        selector = 'pose_2'
        if len(info.keys()) > 4:
            # 2 image condition
            if random.random() > 0.5:
                selector = 'pose_1'

        to_flip = random.random() > 0.5

        ret = {}
        for key, value in info.items():
            if key.startswith(selector):
                continue
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
                
                ret[key.replace('2', '1')] = source

            else:
                # text info
                text = value
                if key != self.text_keys[0]:
                    text = ', '.join([info[self.text_keys[0]], value])
                ret[key.replace('2', '1')] = text
        
        return ret


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('/home/lolicon/data/dataset/blue-archive-test/info.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.keys = [
            "jpg", "txt", "canny_1", "canny_2", "canny_1_text", "canny_2_text"
        ]
        self.path = Path('/home/lolicon/data/dataset/blue-archive-test/')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        info = self.data[idx]
        keys = self.keys

        W, H = 384, 256

        # read image
        image_1 = cv2.imread(str(self.path / "source" / info[keys[2]]))
        image_2 = cv2.imread(str(self.path / "source" / info[keys[3]]))
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

        # resize image
        image_1 = resize_image(image_1, H)
        image_2 = resize_image(image_2, H)

        # annotate
        detected_map_1 = apply_canny(image_1, 100, 200)
        detected_map_2 = apply_canny(image_2, 100, 200)

        assert detected_map_1.shape[1] + detected_map_2.shape[1] <= W
        
        # merge images
        center = np.zeros((H, W - detected_map_1.shape[1] - detected_map_2.shape[1]), dtype="uint8")
        center_image = np.zeros((H, W - detected_map_1.shape[1] - detected_map_2.shape[1], 3), dtype="uint8")
        detected_map_1_ = cv2.hconcat([detected_map_1, center, np.zeros_like(detected_map_2)])
        detected_map_2_ = cv2.hconcat([np.zeros_like(detected_map_1), center, detected_map_2])
        image = cv2.hconcat([image_1, center_image, image_2])

        detected_map_1 = HWC3(detected_map_1_)
        detected_map_2 = HWC3(detected_map_2_)

        with torch.no_grad():
            detected_map_1 = torch.from_numpy(detected_map_1.copy()).float() / 255.0
            # detected_map_1 = einops.rearrange(detected_map_1, 'h w c -> c h w').clone()
            
            detected_map_2 = torch.from_numpy(detected_map_2.copy()).float() / 255.0
            # detected_map_2 = einops.rearrange(detected_map_2, 'h w c -> c h w').clone()
        # print(detected_map_1.shape, detected_map_2.shape)
        ret = {}

        # normalize target to [-1, 1]
        ret["jpg"] = (image.astype(np.float32) / 127.5) - 1.0
        # text prompt
        ret["txt"] = info["txt"]
        ret["canny_1"] = detected_map_1
        ret["canny_2"] = detected_map_2

        ret[keys[4]] = info["txt"] + ', ' + info[keys[4]]
        ret[keys[5]] = info["txt"] + ', ' + info[keys[5]]


        # detected_map = HWC3(detected_map_1)
        # with torch.no_grad():
        #     control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        #     control = torch.stack([control for _ in range(num_samples)], dim=0)
        #     control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        # cv2.hconcat([])

        
        return ret




    # def __getitem__(self, idx):
    #     item = self.data[idx]

    #     source_filename = item['source']
    #     target_filename = item['target']
    #     prompt = item['prompt']

    #     source = cv2.imread('./training/fill50k/' + source_filename)
    #     target = cv2.imread('./training/fill50k/' + target_filename)

    #     # Do not forget that OpenCV read images in BGR order.
    #     source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    #     target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    #     # Normalize source images to [0, 1].
    #     source = source.astype(np.float32) / 255.0

    #     # Normalize target images to [-1, 1].
    #     target = (target.astype(np.float32) / 127.5) - 1.0

    #     return dict(jpg=target, txt=prompt, hint=source)



if __name__ == '__main__':

    dataset = CustomDataset('/home/lolicon/data/dataset/lycoris/')
    

    for i in range(min(len(dataset), 5)):
        print(dataset[i])
    