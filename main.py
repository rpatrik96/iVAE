import argparse
import os
import pickle
import sys

import numpy as np
import torch
import yaml
import wandb

from runners import ivae_runner, tcl_runner


def parse():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--config', type=str, default='ivae_sweep.yaml', help='Path to the config file')
    parser.add_argument('--run', type=str, default='run', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, default='', help='A string for documentation purpose')

    parser.add_argument('--n-sims', type=int, default=1, help='Number of simulations to run')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    return parser.parse_args()


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def make_dirs(args):
    os.makedirs(args.run, exist_ok=True)
    args.log = os.path.join(args.run, 'logs', args.doc)
    os.makedirs(args.log, exist_ok=True)
    args.checkpoints = os.path.join(args.run, 'checkpoints', args.doc)
    os.makedirs(args.checkpoints, exist_ok=True)
    args.data_path = os.path.join(args.run, 'datasets', args.doc)
    os.makedirs(args.data_path, exist_ok=True)


def main():
    args = parse()
    make_dirs(args)

    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    new_config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print(new_config)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    if new_config.tcl:
        print("Running TCL")
        r = tcl_runner(args, new_config)
    else:
        print("Running iVAE")
        r = ivae_runner(args, new_config)


    # r = clean_vae_runner(args, new_config)
    fname = os.path.join(args.run,
                         '_'.join([os.path.splitext(args.config)[0], str(args.seed), str(args.n_sims)]) + '.p')
    pickle.dump(r, open(fname, "wb"))

def main_sweep():


    run = wandb.init()


    wandb.config["device"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    np.random.seed(wandb.config.seed)
    torch.manual_seed(wandb.config.seed)

    if wandb.config.tcl:
        print("Running TCL")
        r = tcl_runner(args, wandb.config)
    else:
        print("Running iVAE")
        r = ivae_runner(args, wandb.config)

if __name__ == '__main__':
    args = parse()

    # Set up your default hyperparameters
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"configs/{args.config}")) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)


    sweep_id = wandb.sweep(config, project="ivae")

    wandb.agent(sweep_id, function=main_sweep)
