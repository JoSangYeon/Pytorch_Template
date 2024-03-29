import wandb
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, notebook

from utils import get_each_output, calc_acc

def train(model, device, optimizer, criterion,
          epochs, save_path, train_loader, valid_loader=None):
    """
    :param model: your model
    :param device: your device(cuda or cpu)
    :param optimizer: your optimizer
    :param criterion: loss function
    :param epochs: train epochs
    :param save_path : checkpoint path
    :param train_loader: train dataset
    :param valid_loader: valid dataset
    """
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        sum_loss = sum_acc = 0
        valid_loss = valid_acc = 0
        bs = train_loader.batch_size

        # in notebook
        # pabr = notebook.tqdm(enumerate(train_loader), file=sys.stdout)

        # in interpreter
        pbar = tqdm(train_loader, file=sys.stdout)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            mb_len = len(target)

            optimizer.zero_grad()
            output = model(data)
            output, _ = get_each_output(output)
            loss = criterion(output, target)
            acc = calc_acc(output, target)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            sum_acc += acc

            loss = sum_loss / (batch_idx + 1)
            acc = sum_acc / (batch_idx * bs + mb_len)
            pbar.set_postfix(epoch=f'{epoch}/{epochs}', loss='{:.4f}, acc={:.4f}'.format(loss, acc))
        pbar.close()

        train_loss = sum_loss / (batch_idx + 1)
        train_acc = sum_acc / (batch_idx * bs + mb_len)

        wandb.log({'train_loss': train_loss,
                   'train_acc': train_acc},
                  step=epoch)

        if valid_loader is not None:
            valid_loss, valid_acc = evaluate(model, device, criterion, valid_loader)

            wandb.log({'valid_loss': valid_loss,
                       'valid_acc': valid_acc},
                      step=epoch)
        print()

        torch.save({
            'model_state_dict': model.module.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'valid_loss': valid_loss,
            'valid_acc': valid_acc,
        }, os.path.join(save_path, f'checkpoint_{epoch}.tar'))

    return model


def evaluate(model, device, criterion, data_loader):
    """
    :param model: your model
    :param device: your device(cuda or cpu)
    :param criterion: loss function
    :param data_loader: valid or test Datasets
    :return: (valid or test) loss and acc
    """
    model.eval()
    sum_loss = sum_acc = 0
    bs = data_loader.batch_size

    with torch.no_grad():
        # in notebook
        # pabr = notebook.tqdm(enumerate(valid_loader), file=sys.stdout)

        # in interpreter
        pbar = tqdm(data_loader, file=sys.stdout)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            mb_len = len(target)

            output = model(data)
            output, _ = get_each_output(output)
            loss = criterion(output, target)
            acc = calc_acc(output, target)

            sum_loss += loss.item()
            sum_acc += acc

            loss = sum_loss / (batch_idx + 1)
            acc = sum_acc / (batch_idx * bs + mb_len)
            pbar.set_postfix(loss='{:.4f}, acc={:.4f}'.format(loss, acc))
        pbar.close()

    total_loss = sum_loss / (batch_idx + 1)
    total_acc = sum_acc / (batch_idx * bs + mb_len)

    return total_loss, total_acc


def main():
    pass


if __name__ == "__main__":
    main()
