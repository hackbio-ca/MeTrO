import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
import pandas as pd

class BimodalDataset(Dataset):

    def __init__(self):


        assert len(tr_data) == len(me_data), "Both modalities must have same number of samples"
        
    def __len__(self) -> int:
        return len(self.modality1_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        mod1 = self.modality1_data[idx]
        mod2 = self.modality2_data[idx]
        label = self.labels[idx] if self.labels is not None else None
        
        return mod1, mod2, label
    
    def load_data(csv_fp, rows_are_samples=True):
        return torch.tensor(pd.read_csv(csv_fp).values, dtype=torch.float32)
    