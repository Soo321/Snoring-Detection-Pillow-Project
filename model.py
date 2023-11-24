import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

class ClassificationNet(nn.Module):
    def __init__(self, category):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size = 3, padding = 1)
        self.resnet18 = models.resnet18(pretrained = False)
        self.fc1 = nn.Linear(1000, 256)
        self.fc2 = nn.Linear(256, category)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet18(x)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ClassificationModel(nn.Module):
    def __init__(self, category):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d()
        )

def make_model(category):
    model = ClassificationNet(category)
    return model

