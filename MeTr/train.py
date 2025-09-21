from MeTr.model import VAE
from MeTr.dataset import BimodalDataset
from MeTr.losses import mixed_vae_loss
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

def training_loop(model, train_loader, val_loader, optimizer, config_d):
    beta = config_d['beta']
    num_epochs = config_d['num_epochs']
    lambda_tr = config_d['lambda_tr']
    lambda_me = config_d['lambda_me']

    log_d = {
        'train_loss': [],
        'train_tr': [],
        'train_me': [],
        'train_kl': [],
        'val_loss': [],
        'val_tr': [],
        'val_me': [],
        'val_kl': [],
    }

    for i in range(num_epochs):
        curr_epoch = i + 1
        train_loss_d = _train_epoch(model, train_loader, optimizer, curr_epoch, beta, lambda_tr, lambda_me)
        val_loss_d = _val_epoch(model, val_loader, curr_epoch, beta, lambda_tr, lambda_me)

        for k, v in train_loss_d.items():
            log_d['train_' + k].append(v)
        for k, v in val_loss_d.items():
            log_d['val_' + k].append(v)

    return log_d

def _train_epoch(model: VAE, train_loader: DataLoader, optimizer, epoch: int, beta: float, lambda1: float, lambda2: float):
    """Training for one epoch"""

    model.train()
    train_loss = 0
    tr_loss = 0
    me_loss = 0
    kl_loss = 0

    N = len(train_loader.dataset)
    
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training dataloader"):

        optimizer.zero_grad()
        out = model(batch)

        loss_d = mixed_vae_loss(out, beta, lambda1, lambda2)
        
        loss = loss_d['loss']
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        tr_loss += loss_d['tr'].item()
        me_loss += loss_d['me'].item()
        kl_loss += loss_d['kld'].item()
    
    avg_loss = train_loss / N
    avg_tr = tr_loss / N
    avg_me = me_loss / N
    avg_kl = kl_loss / N

    print(f'> Epoch: {epoch} Train loss: {avg_loss:.4f}')
    return {
        'loss': avg_loss,
        'tr': avg_tr,
        'me': avg_me,
        'kl': avg_kl
    }

def _val_epoch(model: VAE, val_loader: DataLoader, epoch: int, beta, lambda1, lambda2):
    """Validation for one epoch"""

    model.eval()
    val_loss = 0
    tr_loss = 0
    me_loss = 0
    kl_loss = 0

    N = len(val_loader.dataset)
    
    with torch.no_grad():
        
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validation dataloader"):
            # Forward pass
            out = model(batch)
            
            loss_d = mixed_vae_loss(out, beta, lambda1, lambda2)
            loss = loss_d['loss']
            
            val_loss += loss.item()
            tr_loss += loss_d['tr'].item()
            me_loss += loss_d['me'].item()
            kl_loss += loss_d['kld'].item()
    
    avg_loss = val_loss / N
    avg_tr = tr_loss / N
    avg_me = me_loss / N
    avg_kl = kl_loss / N

    print(f'> Epoch: {epoch} Validation loss: {avg_loss:.4f}')
    return {
        'loss': avg_loss,
        'tr': avg_tr,
        'me': avg_me,
        'kl': avg_kl
    }