import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_config

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.config = config