import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models

NUM_CLASSES = 3

# -------- ResNet18 --------
def get_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Identity()
    return model

# -------- ResNet10 (light version) --------
def get_resnet10():
    model = models.resnet18(weights=None)
    model.layer3 = nn.Identity()
    model.layer4 = nn.Identity()
    model.fc = nn.Identity()
    return model

# -------- SimCLR --------
class SimCLR(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

        device = next(encoder.parameters()).device

        was_training = encoder.training
        encoder.eval()

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(device)
            out = encoder(dummy)
            feat_dim = out.shape[1]

        encoder.train(was_training)

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)

# -------- Classifier --------
class Classifier(nn.Module):
    def __init__(self, encoder : nn.Module):
        super().__init__()
        self.encoder = encoder

        device = next(encoder.parameters()).device

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(device)
            out = encoder(dummy)
            feat_dim = out.shape[1]

        self.fc = nn.Linear(feat_dim, 3)

    def forward(self, x):
        return self.fc(self.encoder(x))