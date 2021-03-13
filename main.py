from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from src.progressive_trainer import ProgressiveGAN


def main():
    name = "celeba"
    art_type = "celeba"
    n = 1000

    gan = ProgressiveGAN(
        iterations=[10000 * n for n in range(0, 6, 1)],
        name=name,
        art_type=art_type,
        n=n,
        batch_sizes=[2 ** n for n in range(6, 0, -1)],
        display_interval=50
    )
    trainer = Trainer(
        auto_select_gpus=True,
        max_epochs=5,
        gpus=1,
        callbacks=[ModelCheckpoint(filepath="checkpoints/latest/", save_last=True)]
    )
    trainer.fit(gan)


if __name__ == '__main__':
    main()
