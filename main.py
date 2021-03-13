from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import json
from src.models.dcgan import DCGAN

from src.general_trainer import CustomModule
from src.progressive_trainer import ProgressiveGAN


def progressive_gan(name: str = "celeba", art_type: str = "celeba", n: int = 100000) -> LightningModule:
    gan = ProgressiveGAN(
        iterations=[100000] * 6,
        name=name,
        art_type=art_type,
        n=n,
        batch_sizes=[8] * 6,
        display_interval=1000,
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

    checkpoint_filepath = json.load(open("src/settings.json"))["filepaths"]["model save path"]
    trainer = Trainer(
        auto_select_gpus=True,
        max_epochs=5,
        gpus=1,
        callbacks=[ModelCheckpoint(filepath=checkpoint_filepath, save_top_k=None, monitor=None)]
    )
    trainer.fit(gan)


if __name__ == '__main__':
    main()
