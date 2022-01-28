import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, notebook


def calc_acc(output, label):
    o_val, o_idx = torch.max(output, dim=-1)
    l_val, l_idx = torch.max(label, dim=-1)
    return (o_idx == l_idx).sum().item()


def train(model, device, optimizer, criterion, epochs, train_loader, valid_loader=None) -> dict:
    """
    :param model: your model
    :param device: your device(cuda or cpu)
    :param optimizer: your optimizer
    :param criterion: loss function
    :param epoch: train epochs
    :param train_loader: train dataset
    :param valid_loader: valid dataset
    :return: history dictionary that contains train_loss, train_acc, valid_loss, valid_acc as list
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': []
    }
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = train_acc = 0

        # in notebook
        # pabr = notebook.tqdm(enumerate(train_loader), file=sys.stdout)

        # in interpreter
        pbar = tqdm(enumerate(train_loader), file=sys.stdout)
        for batch_idx, (data, target) in pbar:
            data, target == data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            acc = calc_acc(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc

            acc = train_acc / (batch_idx * train_loader.batch_size + len(data))
            pbar.set_postfix(epoch=f'{epoch}/{epochs}', loss='{:.6f}, acc={:.3f}'.format(loss, acc))
        pbar.close()

        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_loader.dataset)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        if valid_loader is not None:
            valid_loss, valid_acc = evaluate(model, device, criterion, valid_loader)

            history['valid_loss'].append(valid_loss)
            history['valid_acc'].append(valid_acc)

    return history


def evaluate(model, device, criterion, data_loader):
    """
    :param model: your model
    :param device: your device(cuda or cpu)
    :param criterion: loss function
    :param data_loader: valid or test Datasets
    :return: (valid or test) loss and acc
    """
    model.eval()
    total_loss = total_acc = 0

    with torch.no_grad():
        # in notebook
        # pabr = notebook.tqdm(enumerate(valid_loader), file=sys.stdout)

        # in interpreter
        pbar = tqdm(enumerate(data_loader), file=sys.stdout)

        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device),

            output = model(data)
            loss = criterion(output, target)
            acc = calc_acc(output, target)

            total_loss += loss.item()
            total_acc += acc

            acc = total_acc / (batch_idx * data_loader.batch_size + len(data))
            pbar.set_postfix(loss='{:.6f}, acc={:.3f}'.format(loss, acc))
        pbar.close()

    total_loss = total_loss / len(data_loader)
    total_acc = total_acc / len(data_loader.dataset)

    return total_loss, total_acc


def main():
    pass


if __name__ == "__main__":
    main()
