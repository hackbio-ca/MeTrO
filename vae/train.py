from model import VAE
from dataset import BimodalDataset
from dataloader import BimodalDataSplit
from losses import vae_loss, beta_vae_loss

def train_vae(model, train_loader, optimizer, config_d):

    beta = config_d['beta']
    num_epochs = config_d['num_epochs']

    """Training function for one epoch"""
    model.train()
    train_loss = 0
    
    for i, (batch, _) in enumerate(train_loader):
        batch = batch.to(model.device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(batch)
        if beta is None:
            loss = vae_loss(recon_batch, batch, mu, logvar, model.input_dim) # TODO: check what to actually put for input dim here!!!!
        else:
            loss = beta_vae_loss(recon_batch, batch, mu, longvar, )
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if i % 100 == 0:
            print(f'Train Epoch: {epoch} [{i * len(batch)}/{len(train_loader.dataset)} '
                  f'({100. * i / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(batch):.6f}')
    
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss