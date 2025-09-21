import torch.nn.functional as F
import torch

def bimodal_beta_vae_loss(recon_x1, recon_x2, x1, x2, mu, logvar, config_d):

    total_loss = 0
    # Reconstruction loss for modality 1
    if x1 is not None:
        x1_flat = x1.view(x1.size(0), -1)
        bce_1 = F.binary_cross_entropy(recon_x1, x1_flat, reduction='sum')
        total_loss += config.lambda_1 * bce_1
    
    # Reconstruction loss for modality 2  
    if x2 is not None:
        x2_flat = x2.view(x2.size(0), -1)
        bce_2 = F.binary_cross_entropy(recon_x2, x2_flat, reduction='sum')
        total_loss += config.lambda_2 * bce_2
    
    # KL divergence loss (same regardless of modalities)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return total_loss + config.beta * kld