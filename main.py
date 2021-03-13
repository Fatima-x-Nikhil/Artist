from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models import DCGAN

from src.general_trainer import CustomModule
from src.progressive_trainer import ProgressiveGAN


def progressive_gan(name: str = "celeba", art_type: str = "celeba", n: int = 100000) -> LightningModule:
    gan = ProgressiveGAN(
        iterations=[10000 * n for n in range(1, 7, 1)],
        name=name,
        art_type=art_type,
        n=n,
        batch_sizes=[2 ** n for n in range(6, 0, -1)],
        display_interval=50
    )
    return gan


def dcgan(name: str = "celeba", art_type: str = "celeba", n: int = 100000) -> LightningModule:
    gan = CustomModule(
        models=DCGAN(name, art_type, img_shape=(3, 64, 64)),
        n=n
    )
    return gan


def main():
    gan = progressive_gan("celeba", "celeba", 10000)

    trainer = Trainer(
        auto_select_gpus=True,
        max_epochs=5,
        gpus=1,
        callbacks=[ModelCheckpoint(filepath="checkpoints/latest/", save_last=True)]
    )
    trainer.fit(gan)


if __name__ == '__main__':
    main()
