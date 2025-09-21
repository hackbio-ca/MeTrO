import torch.nn.functional as F
import torch

def beta_vae_loss(recon_tr, recon_me, true_tr, true_me, mu, logvar, beta, lambda_tr, lambda_me):

    # Reconstruction loss for transcripts
    tr_loss = F.mse_loss(recon_tr, true_tr, reduction='mean')
    
    # Reconstruction loss for metabolites
    me_loss = torch.mean(torch.exp(recon_me) - true_me * recon_me)
    
    # KL divergence loss
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    loss_d = {
        'tr': tr_loss,
        'me': me_loss,
        'kl': kl_loss,
        'loss': lambda_tr * tr_loss + lambda_me * me_loss + beta * kl_loss
    }
    return loss_d

def negative_binomial_loss(log_mu, log_theta, targets):
    """Stable Negative Binomial loss"""
    mu = torch.clamp(torch.exp(log_mu), 1e-4, 1e4)
    theta = torch.clamp(torch.exp(log_theta), 0.1, 100)  # Safe theta range
    targets = torch.clamp(targets, 0, 1e4)
    
    try:
        eps = 1e-6
        log_prob = (torch.lgamma(targets + theta + eps) - 
                   torch.lgamma(theta + eps) - 
                   torch.lgamma(targets + 1) +
                   theta * torch.log(theta + eps) + 
                   targets * torch.log(mu + eps) - 
                   (theta + targets) * torch.log(theta + mu + eps))
        
        loss = -torch.mean(torch.clamp(log_prob, -50, 50))
        return loss if not torch.isnan(loss) else torch.tensor(0.0, requires_grad=True)
    except:
        return torch.tensor(0.0, requires_grad=True)

def lognormal_loss(mu, log_sigma, targets):
    """Stable Log-Normal loss"""
    sigma = torch.clamp(torch.exp(log_sigma), 1e-4, 10)
    targets = torch.clamp(targets, 1e-6, 1e6)
    
    try:
        log_targets = torch.log(targets)
        log_prob = (-0.5 * torch.log(2 * torch.pi * sigma**2) - log_targets - 
                   0.5 * ((log_targets - mu) / sigma)**2)
        
        loss = -torch.mean(torch.clamp(log_prob, -50, 50))
        return loss if not torch.isnan(loss) else torch.tensor(0.0, requires_grad=True)
    except:
        return torch.tensor(0.0, requires_grad=True)

def mixed_vae_loss(batch, beta=1.0, lambda_m=1.0, lambda_t=1.0):
    # Extract data and parameters
    true_me = batch['met']
    true_tr = batch['tra']
    
    metab_log_mu = batch['recon_met_log_mu']
    metab_log_theta = batch['recon_met_log_theta']
    trans_mu = batch['recon_tr_mu']
    trans_log_sigma = batch['recon_tr_log_sigma']
    
    mu = batch['mu']
    logvar = batch['logvar']
    
    # Reconstruction losses using proper distributions
    metab_recon_loss = negative_binomial_loss(metab_log_mu, metab_log_theta, true_me)
    trans_recon_loss = lognormal_loss(trans_mu, trans_log_sigma, true_tr)
    
    # KL divergence loss (unchanged)
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    # Total loss
    total_loss = (lambda_m * metab_recon_loss + 
                  lambda_t * trans_recon_loss + 
                  beta * kl_loss)
    
    return {
        'loss': total_loss,
        'me': metab_recon_loss,
        'tr': trans_recon_loss,
        'kl': kl_loss
    }