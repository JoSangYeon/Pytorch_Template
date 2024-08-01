# https://github.com/HideOnHouse/TorchBase

import os
import sys
import json
import wandb
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, AutoModel
from transformers import AutoProcessor

from model import *
from learning import *
from dataset import *
from config import *
from utils import *

import warnings
warnings.filterwarnings(action='ignore')

def setup(rank, world_size, port, SEED):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    set_SEED(SEED)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, port, args):
    # set up #
    setup(rank, world_size, port, args.SEED)

    # get args and config #
    IS_BASE = args.IS_BASE

    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate

    config = BASE_CONFIG() if IS_BASE else LARGE_CONFIG()
    save_path = make_dirs(os.path.join('model', args.model_name))

    # Loading Dataset #
    base_path = os.path.join('..', '..', '_DATASET', 'VALOR-32K')
    train_path = os.path.join(base_path, 'data_train.csv')
    valid_path = os.path.join(base_path, 'data_valid.csv')
    test_path = os.path.join(base_path, 'data_test.csv')

    train_data = pd.read_csv(train_path); train_data = train_data.reset_index()
    valid_data = pd.read_csv(valid_path); valid_data = valid_data.reset_index()
    test_data  = pd.read_csv(test_path);  test_data  = test_data.reset_index()

    train_dataset = MyDataset(train_data, base_path)
    valid_dataset = MyDataset(valid_data, base_path)
    test_dataset  = MyDataset(test_data, base_path)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)
    # test_sampler  = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4 * world_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=4 * world_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4 * world_size)

    # wandb init #
    if args.is_wandb: init_wandb(rank, config=vars(args))

    # Load Training Model #
    model = MyModel(config).to(rank) #; model = model.apply(initialize_weights)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])# find_unused_parameters=True)

    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-12, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    # scheduler = CosineAnnealingWarmUpRestarts(opt, T_0=40, T_mult=2, eta_max=lr, T_up=8, gamma=0.95)

    # Train #
    if rank == 0: print("============================= TRAIN =============================")
    _, history_lst = train(args, rank, model, optimizer, criterion, epochs, save_path,
                           train_loader=train_loader, train_sampler=train_sampler,
                           valid_loader=valid_loader, valid_sampler=valid_sampler)

    # TEST #
    if rank == 0:
        print("============================= TEST =============================")
        test_history, test_factors = evaluate(args, rank, model, criterion, test_loader, mode='test')
        history_lst['test'] = test_history
        print(test_factors)

    # save log(history) #
    log_save_path = make_dirs(os.path.join(save_path, 'history'))
    log_file_name = os.path.join(log_save_path, f'history_gpu_{rank}.json')
    with open(log_file_name, 'w') as json_file:
        json.dump(history_lst, json_file, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch Template')
    parser.add_argument('--project_name', type=str, default='Pytorch_Template', help='Wandb Project name')
    parser.add_argument('--model_name', type=str, default='Baseline_01', help='Model name')
    parser.add_argument('--is_wandb', type=str2bool, default=True, help='Wandb on/off')

    parser.add_argument('--SEED', type=int, default=17, help='Random Seed')
    parser.add_argument('--WORLD_SIZE', type=int, default=2, help='number of distributed processes')
    parser.add_argument('--PORT', type=str, default='12354', help='number of Master PORT Number')
    parser.add_argument('--IS_BASE', type=str2bool, default=True, help='Model Size : True is "BASE" | False is "Large"')

    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=256, help='your batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Max Sequence Length for TextModel')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.spawn(main, args=(args.WORLD_SIZE, args.PORT, args), nprocs=args.WORLD_SIZE, join=True)
    """
CUDA_VISIBLE_DEVICES=2,3 python main_pt.py --SEED 17 --WORLD_SIZE 2 --PORT 12343 --IS_BASE True --mask_ratio 0.6 --learning_rate 5e-4
CUDA_VISIBLE_DEVICES=2,3 python main_pt.py --SEED 17 --WORLD_SIZE 2 --PORT 12347 --IS_BASE False --mask_ratio 0.6 --learning_rate 5e-4

    """