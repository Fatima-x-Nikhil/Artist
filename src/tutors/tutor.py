from torch import nn
from torch.nn import Module
import numpy as np
from typing import Tuple


class Tutor(Module):
    def __init__(self, img_shape: Tuple[int, int, int], name: str, art_type: str):
        self.name = name
        self.art_type = art_type
        self.img_shape = img_shape
        super().__init__()

    def forward(self, _input):
        raise NotImplementedError


