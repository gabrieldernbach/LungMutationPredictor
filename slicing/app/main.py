import base64
import io
from dataclasses import dataclass
from uuid import uuid4

import duckdb
import fsspec
import matplotlib
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed

from toolkit import slice_wsi

matplotlib.use('Agg')
duckdb.register_filesystem(fsspec.filesystem("gs"))


@dataclass
class JobConfig:
    base_path: str = 'gs://bucket_name/iteration_n/data'
    extractor_uuid: str = "a5cd2238-9acf-4e72-b802-42ed1e720a49"
    patch_size: int = 224
    requested_mpp: float = 0.5


def encode_image(buffer):
    """Encodes an image buffer to base64."""
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def decode_image(image_data):
    """Decode a base64 image string into an image object."""
    return Image.open(io.BytesIO(base64.b64decode(image_data)))


def process_job(job, config: JobConfig):
    try:
        """Process a single job."""
        tiles, tissue_boundary_thumbnail = slice_wsi(
            cloud_path=job.wsi_artifact,
            patch_size=config.patch_size,
            requested_mpp=config.requested_mpp,
        )
        df = pd.DataFrame(tiles, columns=["tile_uuid", "x", "y", "tile"])
        df["extractor_uuid"] = config.extractor_uuid
        df["wsi_uuid"] = job.wsi_uuid

        tissue_boundary = {
            "tissue_boundary_uuid": str(uuid4()),
            "wsi_uuid": job.wsi_uuid,
            "tile_extractor": config.extractor_uuid,
            "tissue_boundary_thumbnail": encode_image(tissue_boundary_thumbnail),
        }

        df.to_parquet(f"{config.base_path}/tile/{uuid4()}.parquet")
        pd.Series(tissue_boundary).to_json(f"{config.base_path}/tissue_boundary/{uuid4()}.json")
    except Exception as ex:
        print(f"got error {ex}")


def process_jobs(jobs, config: JobConfig):
    """Process all jobs."""
    ## single-process for debugging
    # for job in jobs.itertuples():
    #     process_job(job, config)
    Parallel(n_jobs=16)(delayed(process_job)(job, config) for job in jobs.itertuples())


def get_job_list(config: JobConfig):
    """Fetch job list that needs processing."""
    if fsspec.filesystem("gs").exists(f'{config.base_path}/tile'):
        stmt = f"""
        SELECT wsi_uuid, wsi_artifact
        FROM read_parquet('{config.base_path}/wsi/*')
        WHERE wsi_uuid NOT IN (
            SELECT DISTINCT(wsi_uuid)
            FROM read_parquet('{config.base_path}/tile/*')
            WHERE extractor_uuid='{config.extractor_uuid}'))
        """
        return duckdb.sql(stmt).df()

    stmt = f"""
    SELECT wsi_uuid, wsi_artifact
    FROM read_parquet('{config.base_path}/wsi/*')
    """
    return duckdb.sql(stmt).df()


if __name__ == "__main__":
    config = JobConfig(
        base_path='gs://bucket_name/iteration_n/data',
        extractor_uuid="a5cd2238-9acf-4e72-b802-42ed1e720a49",
        patch_size=224,
        requested_mpp=0.5,
    )
    jobs = get_job_list(config)
    print(f"found n={len(jobs)} tasks")
    process_jobs(jobs, config)
