from Artist.progressive_trainer import ProgressiveGAN
from pytorch_lightning import Trainer


def train_progressive_gan(settings_filepath: str = "", name: str = "", art_type: str = "", gpu: bool = False):
    gan = ProgressiveGAN(art_type=art_type, name=name, json_path=settings_filepath)
    if gpu:
        trainer = Trainer(checkpoint_callback=True, auto_select_gpus=True, gpus=1)
    else:
        trainer = Trainer(checkpoint_callback=True)
    print("run the following command in your terminal to see the training progress:\n"
          "\ttensorboard --logdir=\"lightning_logs\"")
    trainer.fit(gan)


if __name__ == '__main__':
    train_progressive_gan("../../settings.json", name="Fatima", art_type="celeba", gpu=False)
