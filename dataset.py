import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, *args):
        super(MyDataset, self).__init__()
        self.data = data
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def shape(self):
        return self.data.shape


def main():
    pass

if __name__ == "__main__":
    main()