import glob
import os
import pickle
from pathlib import Path
from typing import Union, Optional, Callable

from torchvision.datasets import DatasetFolder
from tqdm import tqdm

def pkl_loader(path):
    with open(path, 'rb') as f:
        sample = pickle.load(f)
    return sample

class ms_dataset(DatasetFolder):
    def __init__(self, root):
        self.root = self.root
        self.instances = self.make_dataset(root,'.pkl')
        self.loader = pkl_loader

    def __getitem__(self, index: int):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        image = sample[0]
        cond = sample[1]
        return image, cond

    def __len__(self):
        return len(self.instances)

    def make_dataset(self,valid_ext):
        instances=[]
        file_names = glob.glob(os.path.join(self.root, '*'))
        for file_name in file_names:
            if file_name.endswith(valid_ext):
                instances.append(file_name)

        return instances

