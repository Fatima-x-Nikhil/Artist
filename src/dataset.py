from abc import ABC

from pytorch_lightning import LightningDataModule
import json
import os
import json
from tqdm.auto import tqdm
import math
from pathlib import Path
import requests


class UnsplashDownloader:
    def __init__(self):
        json_file = json.load(open("/home/nmelgiri/PycharmProjects/Artist/src/settings.json"))
        self.client_id = json_file["Unsplash"]["Access Key"]
        self.search_url = "https://api.unsplash.com/search/photos/"
        self.urls = {}
        self.path = "/home/nmelgiri/PycharmProjects/images/unsplash"

    def get_image_urls(self, query: str, number_of_urls: int, image_type: str = "small"):
        pages = math.floor(number_of_urls / 30)

        # I bet I'm going to regret this ugly code in the future so here's a cute pic of a cat instead of fixing it
        # https://www.lomsnesvet.ca/wp-content/uploads/sites/21/2019/08/Kitten-Blog-683x1024.jpg
        urls = [
            [image_json["urls"][image_type]
             for image_json in requests.get(
                url=self.search_url,
                params={
                    "client_id": self.client_id,
                    "query": "test",
                    "per_page": 30 if page != pages - 1 else number_of_urls - pages,
                    "page": page
                }
            ).json()["results"]]
            for page in tqdm(range(pages), desc="getting urls for {}".format(query))
        ]

        self.urls.update({query: urls})
        return urls

    def download_urls(self, path: str = ""):
        path = self.path if path == "" else path
        for query in self.urls.keys():
            dir_path = os.path.join(path, query)
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            urls = self.urls[query]
            for index, url in enumerate(urls):
                with open(os.path.join(dir_path, str(index) + ".jpg"), "wb") as file:
                    file.write(requests.get(url, stream=True).content)


class ArtGallery(LightningDataModule, ABC):
    def __init__(self, art_type: str, dirpath: str = ""):
        super().__init__()
        self.dirpath = json.load(open("settings.json"))["filepaths"]["image dirpath"] if dirpath == "" else dirpath
        self.dirpath = os.path.join(self.dirpath, art_type.replace(" ", "_"))



