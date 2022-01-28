import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def show_item(self, idx=0):
        feature, label = self.__getitem__(idx)

        print("Feature's Shape : {}".format(feature.shape))
        print("Label's Shape : {}".format(label.shape))

        return feature, label


def main():
    pass

if __name__ == "__main__":
    main()