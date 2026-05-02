import argparse
import torch
from models import *
from train import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Seed set to {seed}")

def main(mode):
    set_seed(42)
    train_samples, val_samples, test_samples = load_data("./data")

    if mode == "simclr_resnet18":
        encoder = get_resnet18().to(DEVICE)
        simclr = SimCLR(encoder).to(DEVICE)
        pretrain(simclr, train_samples)

        model = Classifier(encoder).to(DEVICE)
        train_classifier(model, train_samples, val_samples, test_samples, "simclr_resnet18")

    elif mode == "sup_resnet18":
        encoder = get_resnet18().to(DEVICE)
        model = Classifier(encoder).to(DEVICE)
        train_classifier(model, train_samples, val_samples, test_samples, "sup_resnet18")

    elif mode == "simclr_resnet10":
        encoder = get_resnet10().to(DEVICE)
        simclr = SimCLR(encoder).to(DEVICE)
        pretrain(simclr, train_samples)

        model = Classifier(encoder).to(DEVICE)
        train_classifier(model, train_samples, val_samples, test_samples, "simclr_resnet10")

    elif mode == "sup_resnet10":
        encoder = get_resnet10().to(DEVICE)
        model = Classifier(encoder).to(DEVICE)
        train_classifier(model, train_samples, val_samples, test_samples, "sup_resnet10")

    else:
        raise ValueError("Invalid mode")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

    main(args.mode)