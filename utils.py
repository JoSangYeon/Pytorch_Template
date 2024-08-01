import os
import wandb
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler

from tqdm import tqdm

def save_model(model, epoch, save_path):
    ckpt_path = os.path.join(save_path, f'checkpoint_{epoch}.tar')
    torch.save({
        'model_state_dict': model.module.state_dict(),
        'epoch': epoch,
    }, ckpt_path)

def get_iter_factor(history, mode='train'):
    factors = {}
    for key, value in history.items():
        factors[f"{mode}/{key}"] = np.mean(value)
    return factors

def get_iter_info(history):
    info = ''
    for key, value in history.items():
        temp = f"{key}: {np.mean(value):.6f} "
        info += temp
    return info[:-1]

def update_history(history, **kwargs):
    for key, value in kwargs.items():
        if key in history.keys():
            history[key].append(value)
        else:
            history[key] = [value]

def init_wandb(rank, config):
    if rank == 0:
        wandb.init(project=config['project_name'])
        wandb.run.name = config['model_name']
        wandb.config.update(config)

def get_each_output(output):
    if type(output) is not list:
        return output
    else:
        return list(map(list, zip(*output)))

def calc_acc(output, label):
    if type(output) is list:  # if multi-gpu settings
        # copy to cpu from gpu
        output = [o.detach().cpu() for o in output]
        output = torch.cat(output, dim=0)
        label = label.detach().cpu()

    o_val, o_idx = torch.max(output, dim=-1)
    l_val, l_idx = torch.max(label, dim=-1)
    return (o_idx == l_idx).sum().item()

def set_SEED(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def make_dirs(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
    return directory

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_name_ext(file_path: str) -> tuple[str, str]:
    """
    :param file_path: absolute or relative file path where file is located
    :return: name, extension
    """
    if os.sep in file_path:
        file_name = file_path.split(os.sep)[-1]
    else:
        file_name = file_path
    if os.extsep in file_name:
        name, ext = file_name.rsplit(os.extsep, maxsplit=1)
    else:
        name, ext = file_name, ""
    return name,


import math
from torch.optim.lr_scheduler import _LRScheduler
# https://gaussian37.github.io/dl-pytorch-lr_scheduler/
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (
                        1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr