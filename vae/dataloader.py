from dataset import BimodalDataset
from torch.utils.data import DataLoader
import torch
from utils import load_config
from typing import Callable

class BimodalDataSplit():

    def __init__(self, dataset, config_d):
        super().__init__()
        self.batch_size = config_d['batch_size']
        self.collate_fn = _custom_collate_fn
        self.train_frac = config_d['train_frac']
        self.val_frac = config_d['val_frac']
        self.dataset = dataset

        assert self.train_frac + self.val_frac == 1

        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [self.train_frac, self.val_frac])

        self.train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(val_dataset, self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def get_train_loader(self):
        return self.train_loader
    
    def get_val_loader(self):
        return self.val_loader

def _custom_collate_fn(samples: list | tuple):

    batch = {}
    for k in samples[0].keys():
        batch[k] = torch.stack([s[k] for s in samples], dim=0)
    
    return batch