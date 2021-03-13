import json
import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule
from src.models import *
from torch.nn.functional import binary_cross_entropy as bce
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

from src.dataset import UnsplashDownloader


class ArtSchool(LightningModule):
    def __init__(self, models: GAN, lr: float = 0.002, b1: float = 0.5, b2: float = 0.999, batch_size: int = 64,
                 dirpath: str = "", train_nontrain: float = 0.8, val_test: float = 0.5, n: int = 1000):
        super().__init__()

        artist, art_critic = models.generator, models.discriminator
        self.artist, self.art_critic = artist.to(self.device), art_critic.to(self.device)
        self.models = models
        self.testing_seed = torch.randn(8, self.models.latent_dim, 1, 1, device=self.device)
        self.batch_size, self.lr, self.b1, self.b2 = batch_size, lr, b1, b2

        # Creating DataLoaders and Configuring Data
        dirpath = self.__unsplash_downloader__(dirpath=dirpath, n=n)
        self.dataset = ImageFolder(
            root=dirpath,
            transform=transforms.Compose([
                transforms.Resize(size=(self.models.img_shape[1], self.models.img_shape[2])),
                transforms.ToTensor()
            ])
        )

    def __unsplash_downloader__(self, dirpath, n):
        dirpath = json.load(open("src/settings.json"))["filepaths"]["image dirpath"] if dirpath == "" else dirpath
        art_type = self.models.art_type
        dirpath = os.path.join(dirpath, art_type.replace(" ", "_"))
        Path(dirpath).mkdir(parents=True, exist_ok=True)
        if len(os.listdir(dirpath)) == 0:
            query = art_type.replace("_", " ")
            print("The art type of {} was not found, downloading a sample of {} images from UnSplash".format(query, n))
            unsplash_downloader = UnsplashDownloader()
            unsplash_downloader.get_image_urls(query=query, number_of_urls=n)
            unsplash_downloader.download_urls(path=dirpath)
        return dirpath

    def training_step(self, batch, batch_idx, optimizer_idx):
        images = batch[0]
        seed = torch.randn(self.batch_size, self.models.latent_dim, 1, 1, device=self.device)

        if optimizer_idx == 0:
            student_artwork = self.artist(seed)
            student_loss = bce(
                self.art_critic(student_artwork).view(-1),
                torch.ones(images.size(0), 1).to(self.device).view(-1)
            )
            self.logger.log_metrics({"{}'s loss".format(self.models.name): student_loss}, self.global_step)
            self.log("{}'s loss".format(self.models.name), student_loss, prog_bar=True)
            return student_loss

        if optimizer_idx == 1:
            real_loss = bce(
                self.art_critic(images).view(-1),
                torch.ones(images.size(0), 1).to(self.device).view(-1)
            )
            fake_loss = bce(
                self.art_critic(self.artist(seed)).view(-1),
                torch.zeros(images.size(0), 1).to(self.device).view(-1)
            )
            tutor_loss = (real_loss + fake_loss) / 2
            self.logger.log_metrics({"{}'s loss".format(self.models.name + " Tutor"): tutor_loss}, self.global_step)
            self.log("Tutor's loss", tutor_loss, prog_bar=True)
            return tutor_loss

    def configure_optimizers(self):
        opt_student = torch.optim.Adam(self.artist.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        opt_tutor = torch.optim.Adam(self.art_critic.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return [opt_student, opt_tutor], []

    def on_epoch_end(self):
        self.logger.experiment.add_image(
            "{}'s artwork of {}".format(self.models.name, self.models.art_type),
            make_grid(self.artist(self.testing_seed.to(self.device))),
            self.current_epoch
        )

    # --------------------------------------------------[DATALOADERS]---------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=12, persistent_workers=True,
                          drop_last=True)
