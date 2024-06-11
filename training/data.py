import io
from dataclasses import dataclass
from pathlib import Path

import duckdb
import fsspec
import numpy as np
import pandas as pd
import torch
import torch.utils.data


@dataclass
class DataLoaders:
    train: torch.utils.data.DataLoader
    train_all: torch.utils.data.DataLoader
    dev: torch.utils.data.DataLoader
    test: torch.utils.data.DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, register):
        self.register = register

    def __len__(self):
        return len(self.register)

    def __getitem__(self, item):
        entry = self.register.iloc[item]
        emb = pd.read_parquet(entry.path, columns=["embedding_uuid", "embedding"])
        emb["embedding"] = emb.embedding.apply(lambda x: np.load(io.BytesIO(x)))
        label = torch.tensor(entry.mutated).unsqueeze(0).float()

        embedding = torch.tensor(np.stack(emb.embedding))
        embedding_uuid = emb.embedding_uuid.tolist()
        return entry.wsi_uuid, embedding, embedding_uuid, label


def parse_hive_partition(base):
    """get the available partitions in a hive-parquet-dataset"""
    gfs = fsspec.filesystem("gs")
    files = [f"gs://{f}" for f in gfs.glob(f"{base}/**/*.parquet")]

    def fun(path):
        parts = Path(path).relative_to(base).parent.parts
        return {k: v for k, v in (part.split("=") for part in parts)}

    partitions = pd.DataFrame([fun(file) for file in files])
    partitions["path"] = [str(Path(file).parent).replace("gs:/", "gs://") for file in files]
    return partitions


def sample_weights(labels):
    labels = torch.tensor(labels).long()
    class_count = torch.bincount(labels)
    class_weighting = 1. / class_count
    return class_weighting[labels]


def get_dataloaders(cfg):
    duckdb.register_filesystem(fsspec.filesystem("gs"))
    # get table [embedder_uuid, wsi_uuid, filepath]
    emb_partitions = parse_hive_partition("gs://bucket_name/iteration_n/data/embedding")
    register = duckdb.sql(f"""
        select mutation.state as mutated,
            wsi.wsi_uuid,
            mutation.fold,
            emb.path,
        from read_parquet("gs://n20_10_lung_hd/ejc_review/data/mutation/*") mutation
        join read_parquet("gs://n20_10_lung_hd/ejc_review/data/wsi/*") wsi on wsi.case_uuid = mutation.case_uuid
        join emb_partitions emb on wsi.wsi_uuid = emb.wsi_uuid
        where variant_calling = {cfg.variant_calling}
        and marker = '{cfg.marker}'
        and wsi.cohort = '{cfg.train_cohort}'
        and emb.embedder_uuid = '{cfg.embedder_uuid}'
        and wsi.quality_control_reject = false
    """).df()

    test = register.query(f"fold=={cfg.test_fold}")
    dev = register.query(f"fold=={cfg.dev_fold}")
    train = register.query(f"(fold!={cfg.dev_fold}) & (fold!={cfg.test_fold})")

    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights(train.mutated.values),
        num_samples=200)

    dataloaders = DataLoaders(
        train=torch.utils.data.DataLoader(
            dataset=Dataset(register=train),
            batch_size=1,
            sampler=train_sampler,
            prefetch_factor=8,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        ),
        train_all=torch.utils.data.DataLoader(
            dataset=Dataset(register=train),
            batch_size=1,
            prefetch_factor=8,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        ),
        dev=torch.utils.data.DataLoader(
            dataset=Dataset(register=dev),
            batch_size=1,
            num_workers=8,
            prefetch_factor=8,
            shuffle=False,
            pin_memory=True,
        ),
        test=torch.utils.data.DataLoader(
            dataset=Dataset(register=test),
            batch_size=1,
            num_workers=8,
            prefetch_factor=8,
            shuffle=False,
            pin_memory=True,
        )
    )
    return dataloaders


def get_holdout(cfg):
    duckdb.register_filesystem(fsspec.filesystem("gs"))
    df = parse_hive_partition("gs://bucket_name/iteration_n/data/embedding")
    holdout = duckdb.sql(f"""
        select mutation.state as mutated,
            wsi.wsi_uuid,
            mutation.fold,
            emb.path,
        from read_parquet("gs://bucket_name/iteration_n/data/mutation/*") mutation
        join read_parquet("gs://bucket_name/iteration_n/data/wsi/*") wsi on wsi.case_uuid = mutation.case_uuid
        join df emb on wsi.wsi_uuid = emb.wsi_uuid
        where variant_calling = '{cfg.variant_calling}'
        and marker = '{cfg.marker}'
        and wsi.cohort = '{cfg.holdout_cohort}'
        and emb.embedder_uuid = '{cfg.embedder_uuid}'
        and wsi.quality_control_reject = false
    """).df()

    dl = torch.utils.data.DataLoader(
        dataset=Dataset(register=holdout),
        batch_size=1,
        num_workers=16,
        prefetch_factor=8,
        shuffle=False,
        pin_memory=True,
    )
    return dl
