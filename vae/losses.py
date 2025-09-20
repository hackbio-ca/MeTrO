import torch.nn.functional as F
import torch

def vae_loss(recon_x, x, mu, logvar, input_dim=784):
    """
    VAE loss function combining reconstruction loss and KL divergence
    """
    # Reconstruction loss (binary cross entropy)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
    
    # KL divergence loss
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD