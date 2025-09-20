import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_config

class VAE(nn.Module):
    def __init__(
            self,
            d_input_tr: int,
            d_input_me: int,
            d_hidden: int,
            d_latent: int):
        
        super().__init__()

class Encoder(nn.Module):
    def __init__(self, d_x, d_h, d_l):
        self.enc1 = nn.Linear(d_x, d_h)
        self.enc21 = nn.Linear(d_h, d_l) # mu layer
        self.enc22 = nn.Linear(d_h, d_l) # logvar layer
    
    def encode(self, x):
        h1 = F.relu(self.enc1(x))
        return self.enc21(h1), self.enc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar)
        
class Decoder(nn.Module):
    def __init__(self, d_l, d_h, d_x):
        self.dec1 = nn.Linear(d_l, d_h)
        self.dec2 = nn.Linear(d_h, d_x)

    def decode(self, z):
        h = F.relu(self.dec1(z))
        return torch.sigmoid(self.dec2(h))
