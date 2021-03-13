from torch.nn import Module
from typing import Tuple


class GAN:
    def __init__(self, generator: Module, discriminator: Module, name: str, art_type: str,
                 img_shape: Tuple[int, int, int], latent_dim: int = 100):
        self.name = name
        self.latent_dim = latent_dim
        self.art_type = art_type
        self.img_shape = img_shape
        self.generator = generator
        self.discriminator = discriminator
