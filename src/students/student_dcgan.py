from torch.nn import Module
from torch import nn
from typing import Tuple
from src.students.student import Student
from src.spare_functions import is_power_of_2


class StudentDCGAN(Student):
    def __init__(self, latent_dim: int, img_shape: Tuple[int, int, int], art_type: str, name: str,
                 generator_features: int, layers: int = 8):
        super().__init__(latent_dim, img_shape, art_type, name)
        # https://pytorch.org/tutorials/_images/dcgan_generator.png

        def create_layers(x: int = 8):
            assert is_power_of_2(x), "input of {} is not a power of 2".format(x)
            yield nn.ConvTranspose2d(
                in_channels=self.latent_dim,
                out_channels=generator_features * x,
                kernel_size=(4, 4),
                stride=(1, 1),
                padding=(0, 0),
                bias=False
            )
            yield nn.BatchNorm2d(generator_features * x)
            yield nn.ReLU(True)
            x /= 2
            while float(x) >= 1.0:
                x = int(x)
                yield nn.ConvTranspose2d(
                    in_channels=generator_features * x * 2,
                    out_channels=generator_features * x,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False
                )
                yield nn.BatchNorm2d(generator_features * x)
                yield nn.ReLU(True)
                x /= 2
            yield nn.ConvTranspose2d(
                in_channels=generator_features,
                out_channels=3,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
            )
            yield nn.Tanh()
        self.main = nn.Sequential(*[x for x in create_layers(layers)])
        print(self.main)

    def forward(self, _input):
        return self.main(_input)


if __name__ == '__main__':
    student = StudentDCGAN(100, img_shape=(3, 64, 64), art_type="?", name="DCGAN", generator_features=64)
