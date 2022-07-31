
from abc import abstractmethod
import torch

class TrackerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, image, metadata=None):
        pass

    @abstractmethod
    def save(self, label):
        pass
