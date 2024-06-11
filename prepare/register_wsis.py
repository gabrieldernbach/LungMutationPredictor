import pathlib
from uuid import uuid4

import duckdb
import fsspec
import pandas as pd

gfs = fsspec.filesystem("gs")
base_path = "gs://n20_10_lung_hd/ejc_review/data"
reject = pd.read_csv("gs://n20_10_lung_hd/hd/bad_slides_anonymized.txt", header=None)[0].tolist()

for cohort, stmt in [
    ('hlcc', 'gs://n20_10_lung_hd/hd/wsis/wsis/kos/**/*.tiff',),
    ('tcga', 'gs://n20_10_lung_hd/tcga/wsis/ffpe/luad/*.svs')
]:
    uris = [dict(
        wsi_artifact=f"gs://{path}",
        wsi_uuid=pathlib.Path(path).stem,
        case_uuid=str(uuid4()),
        tissue_preservation="ffpe",
        cohort=cohort
    )
        for path in gfs.glob(stmt)
    ]
    df = pd.DataFrame(uris)

    if cohort == 'tcga':  # for compatibility with coudray labels on TCGA
        df["case_uuid"] = df.wsi_uuid.apply(lambda x: x[:14])
    if cohort == 'hlcc':  # in hlcc all slides are unique cases
        df["case_uuid"] = df.wsi_uuid.str.extract("(.*)_.*")[0]

    df["quality_control_reject"] = df.wsi_uuid.isin(reject)
    df.to_parquet(f"{base_path}/wsi/{uuid4()}.parquet")

duckdb.register_filesystem(fsspec.filesystem("gs"))
df = duckdb.sql(f"""
select *
from read_parquet('{base_path}/wsi/*')
""").df()
print(df)

########## Load Targets ########

files = [
    {"oncogenic": True, "path": "gs://n20_10_lung_hd/metadata/hlcc_ffpe_oncogenic_mutations.parquet"},
    {"oncogenic": True, "path": "gs://n20_10_lung_hd/metadata/tcga_ffpe_oncogenic_mutations.parquet"},
    {"oncogenic": False, "path": "gs://n20_10_lung_hd/metadata/tcga_ffpe_all_mutations.parquet"},
]


def load_flat(entry):
    df = pd.read_parquet(entry["path"])
    marker = df.columns[df.columns.str.contains("fold")].str.extract("fold_(.*)")[0]
    coll = []
    for m in marker:
        sub = df[[m, f"fold_{m}"]].rename({m: "state", f"fold_{m}": "fold"}, axis=1)
        sub["marker"] = m
        sub["variant_calling"] = entry["oncogenic"]
        coll.append(sub)
    flat = pd.concat(coll)
    flat = flat.reset_index().rename({"case_id": "case_uuid"}, axis=1)
    return flat


for file in files:
    df = load_flat(file)
    df.to_parquet("gs://n20_10_lung_hd/ejc_review/data/mutation", partition_cols=[])
