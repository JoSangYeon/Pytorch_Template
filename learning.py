import os
import sys
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, notebook

from utils import *


def train(args, rank, model, optimizer, criterion, epochs, save_path, accum_step=1,
          train_loader=None, train_sampler=None, valid_loader=None, valid_sampler=None):
    """
    :param args: config
    :param rank: gpu number
    :param model: your model
    :param optimizer: optimizer
    :param criterion: objective function
    :param epochs: train epochs
    :param save_path: save_path
    :param accum_step: gradient accumulate
    :param train_loader: train loader
    :param train_sampler: ddp train sampler
    :param valid_loader: valid loader
    :param valid_sampler: ddp valid sampler

    :return: trained model
    """
    model.to(rank)
    history_lst = {}

    for epoch in range(1, epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch) if train_sampler is not None else ''

        history = {
            'loss': [],
            'acc': []
        }

        train_pbar = tqdm(train_loader, file=sys.stdout) if rank == 0 else train_loader
        optimizer.zero_grad()
        for batch_idx, (features, target) in enumerate(train_pbar):
            if rank == 0: train_pbar.set_description(f'Epoch:{epoch+1}/{epochs}')

            features = features.to(rank)
            target = target.to(rank)
            mb_len = len(target)

            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, target)
            acc = calc_acc(output, target)

            loss /= accum_step
            loss.backward()

            if (batch_idx + 1) % accum_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            history = update_history(history, loss=loss.item(), acc=acc)
            if rank == 0:
                info = get_iter_info(history); train_pbar.set_postfix_str(info)
                if args.is_wandb:
                    factors = get_iter_factor(history, mode='train')
                    wandb.log(factors)
        # 필요시 마지막에 남은 그래디언트를 처리
        if (batch_idx + 1) % accum_step == 0:
            optimizer.step()
            optimizer.zero_grad()
        if rank == 0: train_pbar.close()
        history_lst[f'{epoch}_train'] = history

        if valid_loader is not None:
            valid_sampler.set_epoch(epoch) if valid_sampler is not None else ''
            valid_history, valid_factors = evaluate(args, rank, model, criterion, valid_loader, mode='valid')
            history_lst[f'{epoch}_valid'] = valid_history
            if rank == 0 and args.is_wandb:
                wandb.log(valid_factors)

        # save model #
        if rank == 0: save_model(model, epoch, save_path)

    return model, history_lst


def evaluate(args, rank, model, criterion, data_loader, mode='valid'):
    """
    :param args: config
    :param rank: gpu number
    :param model: your model
    :param criterion: loss function
    :param data_loader: valid or test Datasets

    :return: (valid or test) metric history
    """
    model.eval()
    bs = data_loader.batch_size

    with torch.no_grad():
        history = {
            'loss': [],
            'acc': []
        }

        pbar = tqdm(data_loader, file=sys.stdout) if rank == 0 else data_loader
        for batch_idx, (features, target) in enumerate(pbar):
            if rank == 0: pbar.set_description(f'Valid:')

            features = features.to(rank)
            target = target.to(rank)
            mb_len = len(target)

            output = model(features)
            loss = criterion(output, target)
            acc = calc_acc(output, target)

            history = update_history(history, loss=loss.item(), acc=acc)
            if rank == 0:
                info = get_iter_info(history)
                pbar.set_postfix_str(info)
        if rank == 0: pbar.close()

    return history, get_iter_factor(history, mode=mode)



def main():
    pass


if __name__ == "__main__":
    main()
