import json
import os
from typing import Union, List

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from pytorch_lightning import LightningModule
from torch.nn.functional import binary_cross_entropy as bce
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.datasets import ImageFolder

from src.students.student import Student
from src.tutors.tutor import Tutor
from typing import Tuple


class ArtSchool(LightningModule):
    def __init__(self, student: Student, tutor: Tutor, lr: float = 0.002, b1: float = 0.5, b2: float = 0.999,
                 batch_size: int = 64, dirpath: str = "", train_nontrain_ratio: float = 0.8, val_test_ratio: float = 0.5,
                 img_shape: Tuple[int, int, int] = (3, 64, 64)
                 ):
        super().__init__()

        self.student, self.tutor = student.to(self.device), tutor.to(self.device)
        self.latent_dim, self.img_shape, self.art_type = student.latent_dim, student.img_shape, student.art_type
        self.testing_seed = torch.randn(8, self.latent_dim, 1, 1, device=self.device)
        self.batch_size, self.lr, self.b1, self.b2 = batch_size, lr, b1, b2

        # Creating DataLoaders
        self.dirpath = json.load(open("src/settings.json"))["filepaths"]["image dirpath"] if dirpath == "" else dirpath
        self.dirpath = os.path.join(self.dirpath, self.art_type.replace(" ", "_"))
        dataset = ImageFolder(
            root=self.dirpath,
            transform=transforms.Compose([
                transforms.Resize(size=(self.img_shape[1], self.img_shape[2])),
                transforms.ToTensor()
            ])
        )
        self.training_dataset, self.validation_dataset, self.testing_dataset = random_split(
            dataset=dataset,
            lengths=[
                int(len(dataset) * train_nontrain_ratio),
                int(len(dataset) * (1.0 - train_nontrain_ratio) * val_test_ratio),
                len(dataset) - int(len(dataset) * train_nontrain_ratio) -
                int(len(dataset) * (1.0 - train_nontrain_ratio) * val_test_ratio)
            ]
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        images = batch[0]
        seed = torch.randn(self.batch_size, self.latent_dim, 1, 1, device=self.device)

        # improve the student
        if optimizer_idx == 0:
            student_artwork = self.student(seed)
            # log sampled images
            self.logger.experiment.add_image('generated_images', make_grid(student_artwork[:6]), 0)
            student_loss = bce(
                self.tutor(student_artwork).view(-1),
                torch.ones(images.size(0), 1).to(self.device).view(-1)
            )
            self.log("{}'s loss".format(self.student.name), student_loss, prog_bar=True)
            return student_loss

        # improve the tutor
        if optimizer_idx == 1:
            real_loss = bce(
                self.tutor(images).view(-1),
                torch.ones(images.size(0), 1).to(self.device).view(-1)
            )
            fake_loss = bce(
                self.tutor(self.student(seed)).view(-1),
                torch.zeros(images.size(0), 1).to(self.device).view(-1)
            )
            tutor_loss = (real_loss + fake_loss) / 2
            self.log("{}'s loss".format(self.student.name), tutor_loss, prog_bar=True)
            return tutor_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.student.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.tutor.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        grid = make_grid(self.student(self.testing_seed.type_as(self.student.main[0].weight)))
        self.logger.experiment.add_image("{}'s artwork".format(self.student.name), grid, self.current_epoch)

    # --------------------------------------------------[DATALOADERS]---------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.training_dataset, batch_size=self.batch_size, num_workers=12, persistent_workers=True,
                          drop_last=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=12, persistent_workers=True,
                          drop_last=True)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.testing_dataset, batch_size=self.batch_size, num_workers=12, persistent_workers=True,
                          drop_last=True)

