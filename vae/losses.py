import torch.nn.functional as F
import torch

def vae_loss(recon_x, x, mu, logvar, input_dim):
    """
    VAE loss function combining reconstruction loss and KL divergence
    """
    # Reconstruction loss (binary cross entropy)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
    
    # KL divergence loss
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

def beta_vae_loss(recon_x, x, mu, logvar, input_dim, beta=1.0):
    """
    β-VAE loss function with weighted KL divergence
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weighting factor for KL divergence (β parameter)
        input_dim: Dimensionality of input data
    
    Returns:
        Total loss = Reconstruction Loss + β * KL Divergence
    """
    # Reconstruction loss (binary cross entropy)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
    
    # KL divergence loss
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # b-VAE: weight the KL divergence by beta
    return BCE + beta * KLD