import io

import fsspec
import pandas as pd
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from model import load_model_and_preprocessor


def post_image(image_bytes):
    return requests.post(url="http://localhost:8000/embed_image", data=image_bytes).content


class ParquetDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self, ):
        return len(self.df)

    def __getitem__(self, item):
        entry = self.df.iloc[item]
        img = self.transform(Image.open(io.BytesIO(entry.tile)))
        return img, entry.tile_uuid, entry.wsi_uuid


files = fsspec.filesystem("gs").glob("gs://bucket_name/iteration_n/data/tile/*.parquet")
files = [f"gs://{f}" for f in files]

file = files[0]
tile = pd.read_parquet(file)
model, transform = load_model_and_preprocessor()

ds = ParquetDataset(tile, transform=transform)
dl = DataLoader(ds, batch_size=16, num_workers=2)

for batch in tqdm(dl):
    imgs, tile_uuids, wsi_uuids = batch
    with torch.inference_mode():
        embs = model(imgs.cuda()).cpu()

thread_map(post_image, tile.tile.tolist())
