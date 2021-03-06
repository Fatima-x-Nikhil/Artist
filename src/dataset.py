from abc import ABC

from pytorch_lightning import LightningDataModule
import json
import os
import json
from tqdm.auto import tqdm
import math
from pathlib import Path
import requests
from PIL import Image


class UnsplashDownloader:
    def __init__(self):
        json_file = json.load(open("/home/nmelgiri/PycharmProjects/Artist/src/settings.json"))
        self.client_id = json_file["Unsplash"]["Access Key"]
        self.search_url = "https://api.unsplash.com/search/photos/"
        self.urls = {}
        self.path = "/home/nmelgiri/PycharmProjects/images/unsplash"

    def get_image_urls(self, query: str, number_of_urls: int, image_type: str = "small"):
        pages = math.ceil(number_of_urls / 30)
        urls = []
        for page in tqdm(range(pages), desc="getting urls for {}".format(query)):
            images_data = requests.get(
                url=self.search_url,
                params={
                    "client_id": self.client_id,
                    "query": query,
                    "per_page": 30,
                    "page": page
                }
            )
            for result in images_data.json()["results"]:
                urls.append(result["urls"][image_type])

        self.urls.update({query: urls})
        return urls

    def download_urls(self, path: str = ""):
        path = self.path if path == "" else path
        for query in self.urls.keys():
            dir_path = os.path.join(path, query.replace(" ", "_"))
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            urls = self.urls[query]
            for index, url in enumerate(tqdm(urls, desc="downloading images for {}".format(query))):
                Image.open(requests.get(url, stream=True).raw).save(os.path.join(dir_path, str(index) + ".jpg"))

