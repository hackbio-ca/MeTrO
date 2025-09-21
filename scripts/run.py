from vae.utils import load_config
from vae.dataloader import BimodalDataSplit
from vae.dataset import BimodalDataset
from vae.model import VAE
from vae.train import training_loop
import torch
import pandas as pd
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom_fp', '-c', required=True, help='Path to custom config file (e.g. config/test.yml)')
    parser.add_argument('--default_fp', '-d', type=str, default='config/default.yml', help='Path to default config file')
    args = parser.parse_args()

    # load the config file
    config_d = load_config(args.default_fp, args.custom_fp)

    return config_d

def main():

    # load config dictionary
    config_d = parse_args()
    print('> Configuration settings')
    print(config_d)
    # set seed manually (for reproducibility)
    seed = config_d['seed']
    torch.manual_seed(seed)
    print(f'> Set seed to {seed}')

    # load data
    data_fp = config_d['data_fp']
    print(f'> Loading data from {data_fp}')
    dataset = BimodalDataset(config_d)
    data_module = BimodalDataSplit(dataset, config_d)
    print(f'> Setting up dataloaders')
    train_loader = data_module.get_train_loader()
    val_loader = data_module.get_val_loader()

    # init model
    print(f'> Initializing model')
    model = VAE(d_input_tr=dataset.num_tr,
                d_input_me=dataset.num_met,
                config_d=config_d)
    
    # set up trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=config_d['learning_rate'])

    # set up output dir
    output_dir = os.path.join(config_d['results_dp'], config_d['run_name'])

    if config_d['save_log'] or config_d['save_state']:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    log_fp = os.path.join(output_dir, 'log.csv')
    state_fp = os.path.join(output_dir, 'state_dict.pkl')

    log_d = training_loop(model, train_loader, val_loader, optimizer, config_d)
    print(f'> Training complete')
    print(log_d)
    if config_d['save_log']:
        pd.DataFrame(log_d).to_csv(log_fp)
        print(f'> Saved logs to {log_fp}')
    if config_d['save_state']:
        torch.save(model.state_dict(), state_fp)
        print(f'> Saved state dict to {state_fp}')

if __name__ == '__main__':
    main()