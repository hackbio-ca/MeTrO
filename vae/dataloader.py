from dataset import BimodalDataset
from torch.utils.data import DataLoader
import torch
from utils import load_config

class BimodalDataSplit(DataLoader):

    def __init__(self, dataset, config_d):
        super().__init__()
        self.batch_size = config_d['batch_size']
        self.collate_fn = _custom_collate_fn

def _custom_collate_fn(samples: list | tuple):

    batch = {}
    for k in samples[0].keys():
        batch[k] = torch.stack([s[k] for s in samples], dim=0)
    
    return batch

if __name__ == '__main__':
    
    config_d = load_config('config/default.yml', 'config/test.yml')
    dataset = BimodalDataset(config_d)
    print(dataset)