import hashlib
import logging
import multiprocessing
import os
import uuid
from dataclasses import dataclass
from functools import partial

import duckdb
import fsspec
import pandas as pd
import requests
from tqdm.contrib.concurrent import thread_map

format = '%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(level=logging.INFO, format=format)


def generate_content_uuid(content, namespace=uuid.NAMESPACE_DNS):
    """Generate a UUID based on the SHA-256 hash of the provided content."""
    hash_obj = hashlib.sha256(content.encode('utf-8'))
    content_uuid = uuid.uuid5(namespace, hash_obj.hexdigest())
    return str(content_uuid)


def post_image(url, image_bytes):
    try:
        return requests.post(url=url, data=image_bytes).content
    except requests.RequestException as e:
        logging.error(f"Failed to post images: {e}")
    return None


@dataclass
class JobConfig:
    base_path: str
    extractor_uuid: str
    embedder_uuid: str
    embedder_endpoint: str


def fetch_job_list(config: JobConfig):
    logging.info(f"Fetching jobs")
    duckdb.register_filesystem(fsspec.filesystem("gs"))
    duckdb.sql("set threads=16")
    """Fetch a list of jobs based on the job configuration."""
    try:
        embedding_path = f"{config.base_path}/embedding"
        if fsspec.filesystem("gs").exists(embedding_path):
            query = f"""
            SELECT DISTINCT(wsi_uuid)
            FROM read_parquet('{config.base_path}/tile/*parquet')
            WHERE extractor_uuid = '{config.extractor_uuid}'
            EXCEPT
            SELECT DISTINCT(wsi_uuid)
            FROM read_parquet('{embedding_path}/*parquet')
            WHERE embedder_uuid = '{config.embedder_uuid}'
            """
        else:
            query = f"""
            SELECT DISTINCT(wsi_uuid)
            FROM read_parquet('{config.base_path}/tile/*parquet')
            WHERE extractor_uuid = '{config.extractor_uuid}'
            """
        return duckdb.sql(query).df().wsi_uuid.tolist()
    except Exception as e:
        logging.error(f"Error fetching job list: {e}")
        return []


def fetch_tiles_and_embed(config: JobConfig, wsi_uuid):
    try:
        query = f"""
        SELECT tile_uuid, tile, wsi_uuid
        FROM read_parquet('{config.base_path}/tile/*')
        WHERE wsi_uuid = '{wsi_uuid}'
        AND extractor_uuid = '{config.extractor_uuid}'
        """
        tiles = duckdb.sql(query).df()
        embeddings = thread_map(
            partial(post_image, config.embedder_endpoint),
            tiles.tile.tolist(),
            max_workers=int(os.getenv("NUM_THREADS", 64)),
        )
        return tiles, embeddings
    except Exception as e:
        logging.error(f"Failed to process/embed images for {wsi_uuid}: {e}")
        return pd.DataFrame(), []


def save_embeddings(tiles, embeddings, config: JobConfig):
    """Save embeddings to a Parquet file."""
    try:
        result = tiles[['tile_uuid', 'wsi_uuid']].copy()
        result['embedding_uuid'] = [str(uuid.uuid4()) for _ in range(len(result))]
        result['embedding'] = embeddings
        result['embedder_uuid'] = config.embedder_uuid
        file_uuid = generate_content_uuid(''.join(result.iloc[0][['wsi_uuid', 'embedder_uuid']].values))
        result.to_parquet(f"{config.base_path}/embedding/{file_uuid}.parquet")
    except Exception as e:
        logging.error(f"Failed to save embeddings: {e}")


def worker_init():
    duckdb.register_filesystem(fsspec.filesystem("gs"))
    duckdb.sql("set threads=16")


def execute(job, config):
    print(f"starting job {job}")
    worker_init()
    tiles, embeddings = fetch_tiles_and_embed(config, job)
    has_nan = None in embeddings
    print(f"executed {job}, found nan=={has_nan}")
    if not has_nan:
        save_embeddings(tiles, embeddings, config)


if __name__ == "__main__":
    config = JobConfig(
        base_path=os.getenv("BASE_PATH"),
        extractor_uuid=os.getenv("EXTRACTOR_UUID"),
        embedder_uuid=os.getenv("EMBEDDER_UUID"),
        embedder_endpoint=os.getenv("EMBEDDER_ENDPOINT"),
    )
    jobs = fetch_job_list(config)
    logging.info(f"found {len(jobs)} jobs")
    multiprocessing.set_start_method('spawn')
    with multiprocessing.Pool(processes=int(os.getenv("NUM_PROCESSES")), initializer=worker_init) as pool:
        results = pool.map(partial(execute, config=config), jobs)
