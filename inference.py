import os
import torch
from tqdm import tqdm
from learning import evaluate


def inference(device, criterion, inference_loader):
    file_list = os.listdir(f"models/")

    for file in file_list:
        if file[-2:] != "pt":
            continue

        model = torch.load(file)
        model.to(device); model.eval()
        loss, acc = evaluate(model, device, criterion, inference_loader)
        print("Inference {}".format(file))
        print("\tloss : {:.6f}".format(loss))
        print("\tacc : {:.3f}".format(acc))



def main():
    pass


if __name__ == '__main__':
    main()