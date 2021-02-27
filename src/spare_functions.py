import math
import torch


def is_power_of_2(x: int):
    def log2(y: int):
        return math.log10(y) / math.log10(2)
    return math.ceil(log2(x)) == math.floor(log2(x))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


