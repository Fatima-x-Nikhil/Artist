from torch import nn
from typing import Tuple
from src.tutors.tutor import Tutor
from src.spare_functions import is_power_of_2


class TutorDCGAN(Tutor):
    def __init__(self, img_shape: Tuple[int, int, int], name: str, art_type: str, discriminator_features: int):
        super().__init__(img_shape, name, art_type)

        def create_layers(x: int = 8):
            assert is_power_of_2(x), "input of {} is not a power of 2".format(x)
            y = 1
            yield nn.Conv2d(
                in_channels=3,
                out_channels=discriminator_features,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
            )
            yield nn.LeakyReLU(0.2, inplace=True)
            y *= 2
            while y < 2 * x:
                yield nn.Conv2d(
                    in_channels=discriminator_features * int(y / 2),
                    out_channels=discriminator_features * y,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False
                )
                yield nn.BatchNorm2d(discriminator_features * y)
                yield nn.LeakyReLU(0.2, inplace=True)
                y *= 2
            yield nn.Conv2d(
                in_channels=discriminator_features * int(y / 2),
                out_channels=1,
                kernel_size=(4, 4),
                stride=(1, 1),
                padding=(0, 0),
                bias=False
            )
            yield nn.Sigmoid()
        self.main = nn.Sequential(*[x for x in create_layers(8)])
        print(self.main)

    def forward(self, _input):
        return self.main(_input)


if __name__ == '__main__':
    tutor = TutorDCGAN((3, 64, 64), "?", "?", 64)
