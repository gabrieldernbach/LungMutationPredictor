import duckdb
import fsspec
import pandas as pd
from tqdm import tqdm

from toolbox_heatmapping import tiles_to_wsi, \
    get_heatmap_from_attention, overlay_heatmap, display_thumbnail

gfs = fsspec.filesystem('gs')
duckdb.register_filesystem(gfs)

base_path = "gs://bucket_name/iteration_n/data"


def get_experiment_uuids(marker="EGFR", train_cohort="hlcc"):
    experiment_uuids = duckdb.sql(f"""
    select experiment_uuid
    from read_parquet('{base_path}/experiments.parquet')
    where status == 'success'
    and train_cohort == '{train_cohort}'
    and marker='{marker}'
    """).df().values.flatten().tolist()
    return experiment_uuids


def get_predictions(experiment_uuids):
    experiment = pd.read_parquet(f"{base_path}/experiments.parquet")
    predictions = duckdb.sql(f"""
    select *
    from read_parquet("{base_path}/prediction/*/*") p
    join experiment e on p.experiment_uuid = e.experiment_uuid
    where e.experiment_uuid in {tuple(experiment_uuids)}
    """).df()
    return predictions


def get_wsi_uuids(prediction, limit=50):
    wsi_uuids = duckdb.sql(f"""
    select 
        wsi_uuid, 
        avg(label) as label, 
        avg(prediction) as prediction
    from prediction p
    where p.split in ('test', 'holdout')
    group by wsi_uuid
    order by avg(label) desc, avg(prediction) desc
    limit {limit}
    """).df().wsi_uuid
    return wsi_uuids


def get_attention_for_wsi(prediction, wsi_uuid):
    # for a given slide get the experiments where it was in the test set or hold-out set
    experiment_uuid_subset = (prediction
                              .query(f"wsi_uuid=='{wsi_uuid}'")
                              .query("split in ('test', 'holdout')")
                              .experiment_uuid
                              .tolist())

    # get all the patch attentions for all the runs
    files = [f"{base_path}/explanation/experiment_uuid={euuid}/wsi_uuid={wsi_uuid}" for euuid in experiment_uuid_subset]
    attn = [pd.read_parquet(f) for f in files]
    # average over ensemble
    attn = pd.concat(attn, axis=0).groupby("embedding_uuid").mean().reset_index()

    # fetch embedding_uuid to tile_uuid lookup
    emb = pd.read_parquet(
        f"{base_path}/embedding/embedder_uuid=71291d78-02b3-48ae-bd1a-7e37c012e879/wsi_uuid={wsi_uuid}")
    attn = attn.merge(emb[["embedding_uuid", "tile_uuid"]], on="embedding_uuid")
    return attn


for marker in ["EGFR", "KEAP1", "STK11", "KRAS", "TP53"]:
    # get all EGFR experiments where HLCC was the cohort
    experiment_uuids = get_experiment_uuids(marker="EGFR", train_cohort="hlcc")
    # get all the set of predictions
    prediction = get_predictions(experiment_uuids=experiment_uuids)
    # find the true positive predicted slides
    wsi_uuids = get_wsi_uuids(prediction=prediction, limit=50)


    for wsi_uuid in tqdm(wsi_uuids):
        attn = get_attention_for_wsi(prediction, wsi_uuid)
        tile = pd.read_parquet(f"{base_path}/tile/extractor_uuid=a5cd2238-9acf-4e72-b802-42ed1e720a49/wsi_uuid={wsi_uuid}")

        he_image = tiles_to_wsi(tile)
        heatmap = get_heatmap_from_attention(attn, tile, patch_size=224)
        overlayed = overlay_heatmap(he_image, heatmap)
        display_thumbnail(overlayed)

        from PIL import Image

        img = Image.fromarray(overlayed.resize(0.24).numpy()[..., :-1])
        base_path = 'gs://bucket_name/iteration_n/data'
        fpath = f"{base_path}/heatmaps/marker=EGFR/wsi_uuid={wsi_uuid}/heatmap.jpg"
        with fsspec.open(fpath, "wb") as f:
            img.save(f, format="JPEG")