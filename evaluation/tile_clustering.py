import io
import pickle

import fsspec
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from tqdm.contrib.concurrent import thread_map

gfs = fsspec.filesystem("gs")

base_path = "gs://bucket_name/iteration_n/data"
embedder_uuid = "71291d78-02b3-48ae-bd1a-7e37c012e879"


def load_sample(fpath):
    """ Load and sample embeddings from a given path. """
    embeddings = pd.read_parquet(fpath)
    sampled_embeddings = embeddings.sample(400, replace=True, random_state=0).embedding
    sampled_embeddings = sampled_embeddings.map(lambda x: np.load(io.BytesIO(x)))
    return np.stack(sampled_embeddings)


def fetch_files():
    """ Retrieve file paths from a specified GCS bucket. """
    pattern = f"{base_path}/embedding/embedder_uuid={embedder_uuid}/*"
    files = gfs.glob(pattern)
    return [f"gs://{f}" for f in files]


def perform_kmeans(embeddings):
    """ Fit a MiniBatchKMeans clustering model on the embeddings. """
    kmeans = MiniBatchKMeans(50, verbose=10)
    kmeans.fit(embeddings)
    return kmeans


def save_kmeans_model(kmeans):
    """ Save the trained KMeans model to GCS. """
    fpath = f"{base_path}/clustering/kmeans.pkl"
    with gfs.open(fpath, "wb") as file:
        pickle.dump(kmeans, file)


def save_cluster_centers(kmeans):
    """ Save cluster centers to a Parquet file on GCS. """
    cluster_centers = pd.DataFrame(
        kmeans.cluster_centers_.T,
        columns=[f"cluster_{i}" for i in range(50)]
    )
    fpath = f"{base_path}/clustering/cluster_centers.parquet"
    cluster_centers.to_parquet(fpath)


def deserialize_embeddings(df):
    """ Deserialize embedding data from binary. """
    df['embedding'] = df.embedding.apply(lambda x: np.load(io.BytesIO(x)))
    return df


def assign_cluster(file, kmeans):
    """ Assign clusters to embeddings and return dataframe with assignments. """
    df = pd.read_parquet(file).pipe(deserialize_embeddings)
    wsi_uuid = file.split("/")[-1].split("=")[-1]
    df["cluster_assignment"] = kmeans.predict(np.stack(df.embedding))
    df["wsi_uuid"] = wsi_uuid
    df["embedder_uuid"] = embedder_uuid
    return df[["embedding_uuid", "embedder_uuid", "cluster_assignment", "wsi_uuid"]]


def process_files(files, kmeans):
    """ Process files in parallel and assign clusters. """
    assignments = Parallel(n_jobs=16, verbose=10)(delayed(assign_cluster)(file, kmeans) for file in files)
    combined_assignments = pd.concat(assignments)
    combined_assignments.to_parquet(
        f"{base_path}/cluster_assignment",
        partition_cols=["embedder_uuid"]
    )


if __name__ == "__main__":
    # Main processing sequence
    files = fetch_files()
    embeddings = thread_map(load_sample, files)
    embeddings = np.concatenate(embeddings, 0)

    kmeans = perform_kmeans(embeddings)
    save_kmeans_model(kmeans)
    save_cluster_centers(kmeans)

    process_files(files, kmeans)
