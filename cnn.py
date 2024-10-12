from torch import nn
import torch

def binary(tensor):
    treshold = 0.2
    tensor = torch.gt(tensor, treshold).float()
    return tensor

def binarize(batch):
    new_batch = torch.zeros_like(batch)
    for i in range(len(batch)):
        new_batch[i] = binary(batch[i])
    return new_batch

def norm(tensor):

    mean = torch.mean(tensor, (1, 2))
    std = torch.std(tensor, (1, 2))
    return (tensor-mean)/std

def normalize(batch):
    new_batch = torch.zeros_like(batch)
    for i in range(len(batch)):
        new_batch[i] = norm(batch[i])
    return new_batch

class OldConvNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "CNN"
        self.flatten = nn.Flatten()
        self.normalize = normalize
        self.extractor = nn.Sequential(
            nn.Conv2d(1, 10, 3, 1),
            nn.Conv2d(10, 5, 3, 1),
        )

        self.function_stack = nn.Sequential(
            nn.Linear(5*24*24, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        features = self.extractor(x)
        x = self.flatten(features)
        logits = self.function_stack(x)
        return logits

class ConvNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "CNN"
        self.flatten = nn.Flatten()
        self.normalize = normalize
        self.extractor = nn.Sequential(
            nn.Conv2d(1, 10, 3, 1),
            nn.MaxPool2d(5, 1),
            nn.Conv2d(10, 5, 3, 1),
        )

        self.function_stack = nn.Sequential(
            nn.Linear(5*20*20, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.normalize(x)
        features = self.extractor(x)
        x = self.flatten(features)
        logits = self.function_stack(x)
        return logits
