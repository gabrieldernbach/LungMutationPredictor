import dataclasses
import uuid
from itertools import permutations

import pandas as pd


@dataclasses.dataclass
class Config:
    experiment_uuid: str
    marker: str
    dev_fold: int
    test_fold: int
    train_cohort: str
    holdout_cohort: str
    seed: int
    variant_calling: bool
    status: str
    embedder_uuid: str


if __name__ == "__main__":
    configs = [
        Config(
            experiment_uuid=str(uuid.uuid4()),
            marker=marker,
            dev_fold=dev_fold,
            test_fold=test_fold,
            train_cohort=train_cohort,
            holdout_cohort=holdout_cohort,
            seed=seed,
            variant_calling=variant_calling,
            status="pending",
            embedder_uuid=embedder_uuid
        )
        for marker in ["EGFR", "KEAP1", "STK11", "KRAS", "TP53"]
        for dev_fold, test_fold in permutations(range(5), 2)
        for train_cohort, holdout_cohort in permutations(["tcga", "hlcc"], 2)
        for seed in range(3)
        for variant_calling in [True, False]
        for embedder_uuid in [
            "71291d78-02b3-48ae-bd1a-7e37c012e879",  # UNI
            "364a917b-deff-4aad-ab78-49bce116713e", # CTRANSPATH
        ]
    ]

    pd.DataFrame(configs).to_parquet("gs://bucket_name/iteration_n/data/experiments.parquet", )
