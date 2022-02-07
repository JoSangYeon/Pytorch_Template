import os
import torch
from tqdm import tqdm
from learning import evaluate


def inference(device, criterion, inference_loader):
    file_list = os.listdir(f"models/")

    for file in file_list:
        if file[-2:] != "pt":
            continue

        model = torch.load(f"models/"+file)
        model.to(device); model.eval()

        print("Inference {}".format(file))
        loss, acc = evaluate(model, device, criterion, inference_loader)
        print("\tloss : {:.6f}".format(loss))
        print("\tacc : {:.3f}".format(acc))
        print("\n")



def main():
    pass


if __name__ == '__main__':
    main()