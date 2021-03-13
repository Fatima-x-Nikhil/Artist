from base import GAN
from torch import nn
from typing import Tuple
import warnings


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGAN_Generator(nn.Module):
    def __init__(self, latent_dim: int, features: int = 64):
        super().__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, features * 8, (4, 4), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(features * 8, features * 4, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(features * 4, features * 2, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(features * 2, features, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(features, 3, (4, 4), (2, 2), (1, 1), bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, _input):
        return self.main(_input)


class DCGAN_Discriminator(nn.Module):
    def __init__(self, features: int = 64):
        super(DCGAN_Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, features, (4, 4), (2, 2), (1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(features, features * 2, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(features * 2, features * 4, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(features * 4, features * 8, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(features * 8, 1, (4, 4), (1, 1), (0, 0), bias=False),
            nn.Sigmoid()
        )

    def forward(self, _input):
        return self.main(_input)


class DCGAN(GAN):
    def __init__(self, name: str, art_type: str, img_shape: Tuple[int, int, int],
                 generator_features: int = 64, discriminator_features: int = 64, latent_dim: int = 100):
        super().__init__(
            generator=DCGAN_Generator(latent_dim, generator_features),
            discriminator=DCGAN_Discriminator(discriminator_features),
            name=name,
            art_type=art_type,
            img_shape=img_shape,
            latent_dim=latent_dim
        )
        warnings.warn(
            message="When using DCGAN in this program, only image sizes of 64x64 can work",
            category=UserWarning,
            stacklevel=1
        )

    def get_models(self):
        return self.generator, self.discriminator
