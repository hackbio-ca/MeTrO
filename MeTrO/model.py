import torch
import torch.nn as nn
import torch.nn.functional as F
from MeTrO.losses import beta_vae_loss

class VAE(nn.Module):
    def __init__(
            self,
            d_input_tr: int,
            d_input_me: int,
            config_d):
        d_hidden = config_d['d_hidden']
        d_latent = config_d['d_latent']
        device = config_d['device']
        
        super().__init__()
        self.encoder_m = VarEncoder(d_input_me, d_hidden, d_latent)
        self.encoder_t = VarEncoder(d_input_tr, d_hidden, d_latent)

        self.decoder_m = NBDecoder(d_latent, d_hidden, d_input_me)
        self.decoder_t = LNDecoder(d_latent, d_hidden, d_input_tr)

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
        # Metabolite decoder returns NB parameters
        me_log_mu, me_log_theta = self.decoder_m(z)

        # Transcript decoder returns Log-Normal parameters
        tr_mu, tr_log_sigma = self.decoder_t(z)

        return (tr_mu, tr_log_sigma), (me_log_mu, me_log_theta)
    
    def forward(self, batch):
        x_me = batch['met']
        x_tr = batch['tra']

        mu, logvar = self.encode(x_tr, x_me)

        z = self.reparameterize(mu, logvar)

        (recon_x_tr_mu, recon_x_tr_log_sigma), (recon_x_me_log_mu, recon_x_me_log_theta) = self.decode(z)

        batch['recon_met_log_mu'] = recon_x_me_log_mu
        batch['recon_met_log_theta'] = recon_x_me_log_theta
        batch['recon_tr_mu'] = recon_x_tr_mu
        batch['recon_tr_log_sigma'] = recon_x_tr_log_sigma
        batch['mu'] = mu
        batch['logvar'] = logvar

        return batch
    
class VarEncoder(nn.Module):
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
    
class NBDecoder(nn.Module):
    """Decoder for Negative Binomial distribution (metabolite counts)"""
    def __init__(self, d_latent, d_hidden, d_output):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU()
        )
        # Two heads for NB parameters
        self.mu_head = nn.Linear(d_hidden, d_output)      # log(mean)
        self.theta_head = nn.Linear(d_hidden, d_output)   # log(dispersion)
    
    def forward(self, z):
        h = self.decoder(z)
        log_mu = self.mu_head(h)
        log_theta = self.theta_head(h)
        return log_mu, log_theta

class LNDecoder(nn.Module):
    """Decoder for Log-Normal distribution (transcript RPKM)"""
    def __init__(self, d_latent, d_hidden, d_output):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU()
        )
        # Two heads for Log-Normal parameters
        self.mu_head = nn.Linear(d_hidden, d_output)        # mean in log-space
        self.sigma_head = nn.Linear(d_hidden, d_output)     # log(std) in log-space
    
    def forward(self, z):
        h = self.decoder(z)
        mu = self.mu_head(h)
        log_sigma = self.sigma_head(h)
        return mu, log_sigma
    
if __name__ == '__main__':

    from utils import load_config
    from dataset import BimodalDataset
    from dataloader import BimodalDataSplit

    config_d = load_config('config/default.yml', 'config/test.yml')
    dataset = BimodalDataset(config_d)
    datasplit = BimodalDataSplit(dataset, config_d)
    train_loader = datasplit.get_train_loader()
    val_loader = datasplit.get_val_loader()

    met_d_in = dataset.num_met
    tra_d_in = dataset.num_tr

    model = VAE(tra_d_in, met_d_in, config_d)

    sample_batch = next(iter(train_loader))
    out_batch = model(sample_batch)

    recon_x1, recon_x2, x1, x2, mu, logvar = out_batch['recon_tr'], out_batch['recon_met'], out_batch['tra'], out_batch['met'], out_batch['mu'], out_batch['logvar']
    loss = beta_vae_loss(recon_x1, recon_x2, x1, x2, mu, logvar, 1, 1, 1)
    print(loss)