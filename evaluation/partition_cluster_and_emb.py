import fsspec
import duckdb
from tqdm import tqdm

gfs = fsspec.filesystem("gs")

for _, cluster_idx in tqdm(cid.groupby("wsi_uuid")):
    wsi_uuid = cluster_idx.wsi_uuid[0]
    embedding_base_path = "gs://bucket_name/iteration_n/data/embedding/embedder_uuid=71291d78-02b3-48ae-bd1a-7e37c012e879"
    tile_uuid2embedding_uuid = pd.read_parquet(f"{embedding_base_path}/wsi_uuid={wsi_uuid}", columns=["tile_uuid", "embedding_uuid"])
    tile_base_path = "gs://bucket_name/iteration_n/data/tile/extractor_uuid=a5cd2238-9acf-4e72-b802-42ed1e720a49"
    tile_uuid2tile = pd.read_parquet(f"{tile_base_path}/wsi_uuid={wsi_uuid}", columns=["tile_uuid", "tile"])
    out = duckdb.sql("""
    select cidx.embedding_uuid, cluster_assignment, wsi_uuid, tile
    from cluster_idx cidx
    join tile_uuid2embedding_uuid t2e on cidx.embedding_uuid = t2e.embedding_uuid
    join tile_uuid2tile t on t2e.tile_uuid = t.tile_uuid
    """).df()
    out.to_parquet(
        "gs://bucket_name/iteration_n/data2/clutser_assignment_tiles",
        partition_cols=["cluster_assignment", "wsi_uuid"]
    )