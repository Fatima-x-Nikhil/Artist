import json
import math
import os
from pathlib import Path

import requests
from PIL import Image
from tqdm.auto import tqdm


class UnsplashDownloader:
    def __init__(self, json_path: str = "", image_dir: str = ""):
        self.json_file = json.load(open(json_path))
        self.client_id = self.json_file["Unsplash"]["Access Key"]
        self.search_url = "https://api.unsplash.com/search/photos/"
        self.urls = {}
        self.path = image_dir

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


def unsplash_downloader(art_type, dirpath, n):
    dirpath = json.load(open("src/settings.json"))["filepaths"]["image dirpath"] if dirpath == "" else dirpath
    dirpath = os.path.join(dirpath, art_type.replace(" ", "_"))
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    if len(os.listdir(dirpath)) == 0:
        query = art_type.replace("_", " ")
        print("The art type of {} was not found, downloading a sample of {} images from UnSplash".format(query, n))
        downloader = UnsplashDownloader()
        downloader.get_image_urls(query=query, number_of_urls=n)
        downloader.download_urls(path=dirpath)
    return dirpath
