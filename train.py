import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os, random
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224
BATCH_SIZE = 128
LR = 1e-3
EPOCHS_PRETRAIN = 50
EPOCHS_FINETUNE = 20

# -------- Dataset --------
class TumorDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# -------- Split --------
def load_data(root):
    classes = sorted(os.listdir(root))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    samples = []
    for c in classes:
        for f in os.listdir(os.path.join(root, c)):
            samples.append((os.path.join(root, c, f), class_to_idx[c]))

    random.shuffle(samples)
    n = len(samples)

    return (
        samples[:int(0.7*n)],
        samples[int(0.7*n):int(0.85*n)],
        samples[int(0.85*n):]
    )

# -------- SimCLR Aug --------
class SimCLRTransform:
    def __init__(self):
        self.base = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __call__(self, x):
        return self.base(x), self.base(x)

# -------- Loss --------
def nt_xent_loss(z_i, z_j, temp=0.5):
    N = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)

    sim = torch.matmul(z, z.T) / temp
    mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e4)

    positives = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)])
    negatives = sim[~mask].view(2*N, -1)

    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(2*N, dtype=torch.long, device=z.device)

    return F.cross_entropy(logits, labels)

# -------- Pretrain --------
def pretrain(simclr_model, train_samples):
    loader = DataLoader(
        TumorDataset(train_samples, SimCLRTransform()),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    opt = torch.optim.Adam(simclr_model.parameters(), lr=LR)

    for epoch in range(EPOCHS_PRETRAIN):
        total_loss = 0

        for (x_i, x_j), _ in loader:
            x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)

            z_i = simclr_model(x_i)
            z_j = simclr_model(x_j)

            loss = nt_xent_loss(z_i, z_j)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"[Pretrain] {epoch}: {total_loss/len(loader):.4f}")

# -------- Supervised --------
def train_classifier(model, train_samples, val_samples, test_samples, name):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_loader = DataLoader(
        TumorDataset(train_samples, transform),
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        TumorDataset(val_samples, transform),
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        TumorDataset(test_samples, transform),
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS_FINETUNE):
        model.train()
        correct, total = 0, 0

        # -------- TRAIN --------
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total

        # -------- VALIDATION --------
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                preds = model(x).argmax(1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        print(f"[Epoch {epoch}] Train: {train_acc:.4f} | Val: {val_acc:.4f}")

    # -------- TEST --------
    model.eval()
    test_correct, test_total = 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            preds = model(x).argmax(1)
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)

    test_acc = test_correct / test_total if test_total > 0 else 0.0

    print("\n===== FINAL TEST PERFORMANCE =====")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("=================================\n")

    torch.save(model.state_dict(), f"{name}.pth")
    print(f"Saved {name}.pth")