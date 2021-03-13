import torch
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule
from Artist.models.base import GAN
from torch.nn.functional import binary_cross_entropy as bce
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

from Artist.dataset import unsplash_downloader


class CustomModule(LightningModule):
    def __init__(self, models: GAN, lr: float = 0.002, b1: float = 0.5, b2: float = 0.999, batch_size: int = 64,
                 json_path: str = "", n: int = 1000):
        super().__init__()

        artist, art_critic = models.generator, models.discriminator
        self.generator, self.discriminator = artist.to(self.device), art_critic.to(self.device)
        self.models = models
        self.testing_seed = torch.randn(8, self.models.latent_dim, 1, 1, device=self.device)
        self.batch_size, self.lr, self.b1, self.b2 = batch_size, lr, b1, b2

        # Creating DataLoaders and Configuring Data
        dirpath = unsplash_downloader(self.models.art_type, json_path=json_path, n=n)
        self.dataset = ImageFolder(
            root=dirpath,
            transform=transforms.Compose([
                transforms.Resize(size=(self.models.img_shape[1], self.models.img_shape[2])),
                transforms.ToTensor()
            ])
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        images = batch[0]
        seed = torch.randn(self.batch_size, self.models.latent_dim, 1, 1, device=self.device)

        if optimizer_idx == 0:
            student_artwork = self.generator(seed)
            student_loss = bce(
                self.discriminator(student_artwork).view(-1),
                torch.ones(images.size(0), 1).to(self.device).view(-1)
            )
            self.logger.log_metrics({"{}'s loss".format(self.models.name): student_loss}, self.global_step)
            self.log("{}'s loss".format(self.models.name), student_loss, prog_bar=True)
            return student_loss

        if optimizer_idx == 1:
            real_loss = bce(
                self.discriminator(images).view(-1),
                torch.ones(images.size(0), 1).to(self.device).view(-1)
            )
            fake_loss = bce(
                self.discriminator(self.generator(seed)).view(-1),
                torch.zeros(images.size(0), 1).to(self.device).view(-1)
            )
            tutor_loss = (real_loss + fake_loss) / 2
            self.logger.log_metrics({"{}'s loss".format(self.models.name + " Tutor"): tutor_loss}, self.global_step)
            self.log("Tutor's loss", tutor_loss, prog_bar=True)
            return tutor_loss

    def configure_optimizers(self):
        opt_student = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        opt_tutor = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return [opt_student, opt_tutor], []

    def on_epoch_end(self):
        self.logger.experiment.add_image(
            "{}'s artwork of {}".format(self.models.name, self.models.art_type),
            make_grid(self.generator(self.testing_seed.to(self.device))),
            self.current_epoch
        )

    # --------------------------------------------------[DATALOADERS]---------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=12, persistent_workers=True,
                          drop_last=True)
