from model import VAE
from dataset import BimodalDataset
from torch.utils.data import DataLoader
from losses import beta_vae_loss
import torch
import tqdm

def training_loop(model, train_loader, val_loader, optimizer, config_d):
    beta = config_d['beta']
    num_epochs = config_d['num_epochs']
    lambda1 = config_d['lambda1']
    lambda2 = config_d['lambda2']

    log_d = {
        'train_loss': [],
        'val_loss': []
    }

    for i in range(num_epochs):
        curr_epoch = i + 1
        train_loss = _train_epoch(model, train_loader, optimizer, curr_epoch, beta, lambda1, lambda2)
        val_loss = _val_epoch(model, val_loader, optimizer, curr_epoch, beta, lambda1, lambda2)

        log_d['train_loss'].append(train_loss)
        log_d['val_loss'].append(val_loss)

    return log_d

def _train_epoch(model: VAE, train_loader: DataLoader, optimizer, epoch: int, beta: float, lambda1: float, lambda2: float):
    """Training for one epoch"""

    model.train()
    train_loss = 0
    
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training dataloader"):

        optimizer.zero_grad()
        out = model(batch)
        recon_tr = out['recon_tr']
        recon_met = out['recon_met']
        mu = out['mu']
        logvar = out['logvar']

        loss = beta_vae_loss(recon_tr, recon_met, out['tra'], out['met'], mu, logvar, beta, lambda1, lambda2)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'> Epoch: {epoch} Train loss: {avg_loss:.4f}')
    return avg_loss

def _val_epoch(model: VAE, val_loader: DataLoader, epoch: int, beta, lambda1, lambda2):
    """Validation for one epoch"""

    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validation dataloader"):
            # Forward pass
            out = model(batch)
            recon_tr = out['recon_tr']
            recon_met = out['recon_met']
            mu = out['mu']
            logvar = out['logvar']
            
            loss = beta_vae_loss(recon_tr, recon_met, out['tra'], out['met'], mu, logvar, beta, lambda1, lambda2)
            val_loss += loss.item()
            
    avg_loss = val_loss / len(val_loader.dataset)
    print(f'> Epoch: {epoch} Val loss: {avg_loss:.4f}')
    return avg_loss