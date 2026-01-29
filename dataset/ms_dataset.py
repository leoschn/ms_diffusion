import glob
import os
import pickle
from pathlib import Path
from typing import Union, Optional, Callable

import torch
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from tqdm import tqdm

class CropTransform:
    def __init__(self, top: int, left: int, height: int, width: int):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, image:torch.Tensor) -> torch.Tensor:
        return image[self.top:self.top+self.height, self.left:self.left+self.width]



def pkl_loader(path):
    with open(path, 'rb') as f:
        sample = pickle.load(f)
    return sample

class ms_dataset(DatasetFolder):
    def __init__(self, root, im_size=(512,1024)):
        self.root = root
        self.instances = self.make_dataset('.pkl')
        self.loader = pkl_loader
        self.transform_img = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((1.44), (1.19)),
                                             CropTransform(top=90, left=0, height=422, width=1024),
                                             transforms.Resize(im_size)])
        self.transform_cond = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize(im_size)])

    def __getitem__(self, index: int):

        path = self.instances[index]
        window = int(path.split('_')[-1].split('.')[0])
        sample = self.loader(path)
        image = sample[0]
        cond = sample[1]
        if self.transform_img is not None:
            image = self.transform_img(image)
        if self.transform_cond is not None:
            cond = self.transform_cond(cond)
        return image, cond, path, window

    def __len__(self):
        return len(self.instances)

    def make_dataset(self, valid_ext):
        instances=[]
        file_names = glob.glob(os.path.join(self.root, '*'))
        for file_name in file_names:
            if file_name.endswith(valid_ext):
                instances.append(file_name)

        return instances