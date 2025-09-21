import torch.nn.functional as F
import torch

def beta_vae_loss(recon_tr, recon_me, true_tr, true_me, mu, logvar, beta, lambda_tr, lambda_me):

    # Reconstruction loss for transcripts (Gaussian MSE bc continuous)
    tr_loss = F.mse_loss(recon_tr, true_tr, reduction='mean')
    
    # Reconstruction loss for metabolites (Poisson bc discrete counts)
    me_loss = torch.mean(torch.exp(recon_me) - true_me * recon_me)
    
    # KL divergence loss
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    loss_d = {
        'tr': tr_loss,
        'me': me_loss,
        'kld': kl_loss,
        'loss': tr_loss + me_loss + beta * kl_loss
    }
    return loss_d

def poisson_nll_loss(log_rates, targets):
    rates = torch.exp(log_rates)
    nll = rates - targets * log_rates
    return torch.mean(nll)