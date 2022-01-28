# https://github.com/HideOnHouse/TorchBase

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from dataset import MyDataset
from model import MyModel
from learning import train, evaluate, calc_acc
from inference import inference


def main():
    model = MyModel()
    # train
    train_dataset = MyDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    history = train(model, device, optimizer, criterion, 16, train_dataloader)

    # Test
    test_dataset = MyDataset()
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    evaluate(model, device, test_dataloader, criterion)

    # Inference
    infer_dataset = MyDataset()
    infer_dataloader = DataLoader(infer_dataset, batch_size=32, shuffle=False)
    inference(model, device, infer_dataloader)


if __name__ == '__main__':
    main()