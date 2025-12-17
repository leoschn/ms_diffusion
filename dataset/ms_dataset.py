import glob
import os
import pickle
from pathlib import Path
from typing import Union, Optional, Callable
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from tqdm import tqdm

def pkl_loader(path):
    with open(path, 'rb') as f:
        sample = pickle.load(f)
    return sample

class ms_dataset(DatasetFolder):
    def __init__(self, root):
        self.root = root
        self.instances = self.make_dataset('.pkl')
        self.loader = pkl_loader
        self.transform_img = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((1.44), (1.19)),
                                             transforms.Resize((256,1024))])
        self.transform_cond = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((256,1024))])

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

    def make_dataset(self,valid_ext):
        instances=[]
        file_names = glob.glob(os.path.join(self.root, '*'))
        for file_name in file_names:
            if file_name.endswith(valid_ext):
                instances.append(file_name)

        return instances