import io
import os
from random import sample

import fsspec
import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

for cid in tqdm(range(50)):
    opath = f"cluster_{cid}.jpg"
    if os.path.exists(opath):
        continue

    gfs = fsspec.filesystem("gs")
    base_path = f"gs://bucket_name/iteration_n/data/clutser_assignment_tiles/cluster_assignment={cid}"

    files = gfs.glob(f"{base_path}/*")
    files = [f"gs://{f}" for f in files]
    files = sample(files, k=100)

    tiles = thread_map(lambda x: pd.read_parquet(x).sample(), files)
    df = pd.concat(tiles)
    df['tile'] = df.tile.apply(lambda x: np.array(Image.open(io.BytesIO(x))))
    arr = np.stack(df.tile.values)
    arr_ = rearrange(arr, "(b1 b2) h w c -> (b1 h) (b2 w) c", b1=10)
    img = Image.fromarray(arr_)
    img.save(opath)