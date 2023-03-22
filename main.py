# https://github.com/HideOnHouse/TorchBase

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from dataset import *
from learning import *
from model import get_Model
from inference import inference

def draw_history(history, save_path=None):
    train_loss = history["train_loss"]
    train_acc = history["train_acc"]
    valid_loss = history["valid_loss"]
    valid_acc = history["valid_acc"]

    plt.subplot(2,1,1)
    plt.plot(train_loss, label="train")
    plt.plot(valid_loss, label="valid")
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(train_acc, label="train")
    plt.plot(valid_acc, label="valid")
    plt.legend()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_path, 'train_plot.png'), dpi=300)

def set_device(device_num=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        device += ':{}'.format(device_num)
    return device

def set_save_path(model_name, epochs, batch_size):
    directory = os.path.join('models', f'{model_name}_e{epochs}_bs{batch_size}')
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
    return directory

def main():
    # args
    model_name = "Mymodel"
    epochs = 15
    batch_size = 128
    device = set_device(device_num=0)
    save_path = set_save_path(model_name, epochs, batch_size)

    # Datasets
    train_path = ""
    test_path = ""

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # your Data Pre-Processing
    train_x, train_y = train_data.iloc[:, 1:], train_data.iloc[:, :1]
    test_x, test_y = test_data.iloc[:, 1:], test_data.iloc[:, :1]

    # data split
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, stratify=train_y, random_state=17, test_size=0.1)

    # Check Train, Valid, Test Image's Shape
    print("The Shape of Train Images: ", train_x.shape)
    print("The Shape of Valid Images: ", valid_x.shape)
    print("The Shape of Test Images: ", test_x.shape)

    # Check Train, Valid Label's Shape
    print("The Shape of Train Labels: ", train_y.shape)
    print("The Shape of Valid Labels: ", valid_y.shape)
    print("The Shape of Valid Labels: ", test_y.shape)

    # Create Dataset and DataLoader
    train_dataset = MyDataset(train_x, train_y)
    valid_dataset = MyDataset(valid_x, valid_y)
    test_dataset = MyDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # label_tags
    label_tags = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    # modeling
    model = get_Model(model_name); model.to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # train
    print("============================= Train =============================")
    history = train(model, device, optimizer, criterion, epochs, save_path, train_loader, valid_loader)

    # Test
    print("============================= Test =============================")
    test_loss, test_acc = evaluate(model, device, criterion, test_loader)
    print("test loss : {:.6f}".format(test_loss))
    print("test acc : {:.3f}".format(test_acc))

    # save model
    torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}.pt"))
    with open(os.path.join(save_path, "train_history.pickle"), 'wb') as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)

    # plot history
    draw_history(history)

    # Inference
    # print("=========================== Inference ===========================")
    # inference(device, criterion, infer_dataloader)


if __name__ == '__main__':
    main()