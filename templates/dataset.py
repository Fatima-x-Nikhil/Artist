from abc import ABC

from pytorch_lightning import LightningDataModule
import json
import os


class ArtGallery(LightningDataModule, ABC):
    def __init__(self, art_type: str, dirpath: str = ""):
        super().__init__()
        self.dirpath = json.load(open("settings.json"))["filepaths"]["image dirpath"] if dirpath == "" else dirpath
        self.dirpath = os.path.join(self.dirpath, art_type.replace(" ", "_"))



