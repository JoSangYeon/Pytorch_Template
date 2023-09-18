# https://github.com/HideOnHouse/TorchBase

import os
import wandb
import pickle
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split

from dataset import *
from learning import *
from model import *
from inference import *
from utils import DataParallelModel, DataParallelCriterion
from utils import set_device, set_save_path, draw_history
SEED = 17

def main():
    # Define project
    project_name = ''
    model_name = ''

    wandb.init(project=project_name)
    wandb.run.name = model_name
    # wandb.run.save()

    # args
    epochs = 15
    batch_size = 128
    lr = 1e-3

    main_device, device_ids = set_device(main_device_num=0, using_device_num=4)
    save_path = set_save_path(model_name, epochs, batch_size)

    config = {
        'learning_rate': lr,
        'batch_size': batch_size,
        'epochs': epochs,
    }
    wandb.config.update(config)

    # Datasets
    train_path = ""
    test_path = ""

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    ## your Data Pre-Processing

    ## Create Dataset and DataLoader
    train_dataset = MyDataset(train_data)
    test_dataset = MyDataset(test_data)

    ## data split
    train_dataset, valid_dataset = random_split(train_dataset, [0.8, 0.2],
                                                generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    ## Check Train, Valid, Test Dataset Shape
    print("The Length of Train Data: ", len(train_dataset))
    print("The Length of Valid Data: ", len(valid_dataset))
    print("The Length of Test Data: ", len(test_dataset))

    # label_tags
    label_tags = [] #['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    # modeling
    model = DataParallelModel(MyModel(), device_ids=device_ids)#; model.to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    criterion = DataParallelCriterion(criterion, device_ids=device_ids)

    # train
    print("============================= Train =============================")
    _ = train(model, main_device, optimizer, criterion, epochs, save_path, train_loader, valid_loader)

    # Test
    print("============================= Test =============================")
    test_loss, test_acc = evaluate(model, main_device, criterion, test_loader)
    print("test loss : {:.6f}".format(test_loss))
    print("test acc : {:.3f}".format(test_acc))

    # plot history
    # draw_history(history)

    # Inference
    # print("=========================== Inference ===========================")
    # inference(main_device, criterion, infer_dataloader)


if __name__ == '__main__':
    main()
    # CUDA_VISIBLE_DEVICES=2,3,4,5 python main.py