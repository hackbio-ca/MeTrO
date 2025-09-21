import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(
            self,
            d_input_tr: int,
            d_input_me: int,
            d_hidden: int,
            d_latent: int,
            device: str):
        
        super().__init__()
        self.encoder_m = Encoder(d_input_me, d_hidden, d_latent)
        self.encoder_t = Encoder(d_input_tr, d_hidden, d_latent)
        self.decoder_m = Decoder(d_latent, d_hidden, d_input_me)
        self.decoder_t = Decoder(d_latent, d_hidden, d_input_tr)

        self.joint_layer = nn.Linear(4 * d_latent, d_hidden)
        self.joint_mu = nn.Linear(d_hidden, d_latent)
        self.joint_logvar = nn.Linear(d_hidden, d_latent)

        self.device = torch.device(device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x_tr, x_me):
        mu_tr, logvar_tr = self.encoder_t(x_tr)
        mu_me, logvar_me = self.encoder_m(x_me)

        # collect and fuse into joint latent params
        joint = torch.cat([mu_tr, logvar_tr, mu_me, logvar_me], dim=1)
        h = F.relu(self.joint_layer(joint))
        joint_mu = self.joint_mu(h)
        joint_logvar = self.joint_logvar(h)

        return joint_mu, joint_logvar
    
    def decode(self, z):
        recon_x_me = self.decoder_m(z)
        recon_x_tr = self.decoder_t(z)

        return recon_x_tr, recon_x_me
    
    def forward(self, batch):
        x_me = batch['met']
        x_tr = batch['tra']

        mu, logvar = self.encode(x_tr, x_me)

        z = self.reparameterize(mu, logvar)

        recon_x_tr, recon_x_me = self.decode(z)

        batch['recon_met'] = recon_x_me
        batch['recon_tr'] = recon_x_tr
        batch['mu'] = mu
        batch['logvar'] = logvar

        return batch
class Encoder(nn.Module):
    def __init__(self, d_x, d_h, d_l):
        super().__init__()
        self.enc1 = nn.Linear(d_x, d_h)
        self.enc21 = nn.Linear(d_h, d_l) # mu layer
        self.enc22 = nn.Linear(d_h, d_l) # logvar layer
    
    def forward(self, x):
        h = F.relu(self.enc1(x))
        mu = self.enc21(h)
        logvar = self.enc22(h)
        return mu, logvar
        
class Decoder(nn.Module):
    def __init__(self, d_l, d_h, d_x):
        super().__init__()
        self.dec1 = nn.Linear(d_l, d_h)
        self.dec2 = nn.Linear(d_h, d_x)

    def forward(self, z):
        h = F.relu(self.dec1(z))
        recon_x = torch.sigmoid(self.dec2(h))
        return recon_x