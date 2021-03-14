from Artist.progressive_trainer import ProgressiveGAN

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


iterations = [50000] * 6
# batch_sizes = [2 ** n for n in range(7, 1, -1)]
batch_sizes = [32, 32, 16, 16, 8, 4]

gan = ProgressiveGAN(
    art_type="celeba-hq",
    name="AI_Artist_Type_1",
    json_path="settings.json",
    batch_sizes=batch_sizes,
    n_label=1,
    display_interval=100,
    n=1000,
    iterations=iterations,
    display_length=64,
    initial_step=0,
    gpu=True
)

checkpoint = ModelCheckpoint(save_top_k=None, monitor=None)

trainer = Trainer(
    gpus=1,
    auto_select_gpus=True,
    callbacks=[checkpoint]
)

trainer.fit(gan)
