import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
import pandas as pd

class BimodalDataset(Dataset):

    def __init__(self, config_d):
        self.met_tr_df = pd.read_csv(config_d['data_fp'])
        self.num_samples = len(self.met_tr_df.index)
        mask = self.met_tr_df.columns.str.contains(r'^ENSG') # Split df into met data and tra data
        #self.tr_df = self.met_tr_df.loc[:, mask] # DataFrame with transcriptomics data
        #self.met_df = self.met_tr_df.loc[:, ~mask] # DataFrame with metabolomics data
        self.tr_df = self.met_tr_df.loc[:, self.met_tr_df.columns[mask]] # DataFrame with transcriptomics data
        self.met_df = self.met_tr_df.loc[:, self.met_tr_df.columns[~mask]] # DataFrame with metabolomics data
        self.num_tr = len(self.tr_df.columns)
        self.num_met = len(self.met_df.columns)

        if 'CCLE_ID' in self.met_df.columns:
            self.num_met -= 1
        if 'DepMap_ID' in self.met_df.columns:
            self.num_met -= 1
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        { 
            'name': sample name (ccle_id),
            'met': metabolite data,
            'tra': transcriptomics data
        }
        """

        # Remove the CCLE_ID and DepMap_ID cols
        remove_ccle = False

        if 'CCLE_ID' in self.met_df.columns:
            # Extract the CCLE_ID
            remove_ccle = True

        if 'DepMap_ID' in self.met_df.columns:
            met_df.pop('DepMap_ID')
        
        # Extract desired row
        tr_df = self.tr_df.loc[idx]

        # DataFrame with metabolomics data
        met_df = self.met_df.loc[idx]

        # Extract the CCLE_ID
        ccle_id = met_df['CCLE_ID']

        if remove_ccle:
            met_df.pop('CCLE_ID')

        return {'name': ccle_id, 'met': torch.tensor(list(met_df)).to(torch.float32), 'tra': torch.tensor(list(tr_df)).to(torch.float32)}