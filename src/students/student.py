from torch.nn import Module
from torch import nn
import numpy as np
from typing import Tuple


class Student(Module):
    def __init__(self, latent_dim: int, img_shape: Tuple[int, int, int], art_type: str, name: str):
        self.art_type = art_type
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.name = name
        super().__init__()

    def forward(self, _input):
        raise NotImplementedError
