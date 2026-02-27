import glob
import os
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder

class CropTransform:
    def __init__(self, top: int, left: int, height: int, width: int):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:  # (H, W)
            return image[self.top:self.top + self.height,self.left:self.left + self.width]
        elif image.ndim == 3:  # (C, H, W)
            return image[:,self.top:self.top + self.height,self.left:self.left + self.width]
        else:
            raise ValueError("Unsupported image shape")

class LogTransform:
    def __init__(self):
        pass

    def __call__(self, image : np.ndarray) -> np.ndarray:
        return np.log(image+1)

def pkl_loader(path):
    with open(path, 'rb') as f:
        sample = pickle.load(f)
    return sample

class ms_dataset(DatasetFolder):
    def __init__(self, root, im_size=(512,1024), window = 'all',total_windows = 100):
        self.root = root
        self.window = str(window)
        self.total_windows = total_windows
        self.instances = self.make_dataset('.pkl')
        self.loader = pkl_loader
        # self.transform_img = None
        self.transform_img = transforms.Compose([
                                             CropTransform(top=90, left=0, height=422, width=1024),
                                             transforms.ToTensor(),
                                             transforms.Resize(im_size),
                                             #transforms.Normalize((1.0207), (1.0011)), #std 1 mean 0 but lies in roughly [-1 5] => outside pred range
                                             transforms.Normalize((3.04),(3.04)), #=> fixes the range even if nor more std 1 mean 0
        ])

        # self.transform_cond = None
        self.transform_cond = transforms.Compose([
                                            LogTransform(),
                                            CropTransform(top=90, left=0, height=422, width=1024),
                                            transforms.ToTensor(),
                                            transforms.Resize(im_size),
                                            #normalization does not matter for condition mask ?

        ])

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


        return image, cond, path, window/self.total_windows

    def __len__(self):
        return len(self.instances)

    def make_dataset(self, valid_ext):
        instances=[]
        file_names = glob.glob(os.path.join(self.root, '*'))
        for file_name in file_names:
            if file_name.endswith(valid_ext):
                if self.window =='all':
                    instances.append(file_name)
                elif 'ms2_'+self.window+'.pkl' in file_name:
                    instances.append(file_name)

        return instances