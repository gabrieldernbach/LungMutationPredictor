from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

base_path = "gs://bucket_name/iteration_n/data"


class RocAucScore:
    def __init__(self):
        self.labels = []
        self.predictions = []

    def collect(self, labels, predictions):
        self.labels.extend(labels.flatten().numpy())
        self.predictions.extend(predictions.flatten().detach().numpy())

    def compute(self):
        auc = roc_auc_score(self.labels, self.predictions)
        self.__init__()
        return auc


def fit(learner, data):
    model, optimizer, criterion = learner

    auc = RocAucScore()
    ckpt = model.state_dict().copy()

    best_score = 0
    n_step_no_ckpt = 0

    for epoch in range(200):
        n_step_no_ckpt += 1
        model.train()
        for wsi_uuid, embedding, embedding_uuid, labels in tqdm(data.train, disable=True):
            pred = model(embedding)
            loss = criterion(pred, labels.mul(0.8).add(0.1))
            auc.collect(labels=labels, predictions=pred)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        msg = f"train {auc.compute()=:.3f} "

        model.eval()
        with torch.no_grad():
            for wsi_uuid, embedding, embedding_uuid, labels in tqdm(data.dev, disable=True):
                pred = model(embedding)
                auc.collect(labels=labels, predictions=pred)

        current_score = auc.compute()
        msg += f"dev auc {current_score:.3f} "
        if current_score > best_score:
            ckpt = model.state_dict().copy()
            best_score = current_score
            msg += "saved ckpt"
            n_step_no_ckpt = 0

        print(msg)
        if n_step_no_ckpt >= 5:
            print("early stopping")
            break

    model.load_state_dict(ckpt)
    return model


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def compute_attention_and_prediction(model, embedding):
    """Compute attention and prediction from a model given an embedding."""
    with torch.no_grad():
        attention = model.attn(embedding).squeeze().numpy()
        prediction = model(embedding).squeeze()
        return attention, prediction


def write_to_parquet(df):
    df.to_parquet(
        f"{base_path}/explanation",
        partition_cols=["experiment_uuid", "wsi_uuid"]
    )


def process_embedding_data(model, embedding_uuid, embedding):
    """Process embedding data through the model to compute outcomes and predictions."""
    patch_attention, slide_prediction = compute_attention_and_prediction(model, embedding)
    patch_attention = pd.DataFrame({
        'embedding_uuid': flatten(embedding_uuid),
        'attention': patch_attention,
    })
    return patch_attention, slide_prediction


def predict(model, dataloader, config, split: str):
    """Main processing loop to handle embeddings, compute results, and manage I/O asynchronously."""
    slide_results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for wsi_uuid, embedding, embedding_uuid, labels in tqdm(dataloader, disable=False):
            attention, pred = process_embedding_data(model, embedding_uuid, embedding)
            slide_results.append((wsi_uuid[0], pred.item(), labels[0].item()))
            if split in ["test", "holdout"]:
                attention = attention.assign(
                    wsi_uuid=wsi_uuid[0],
                    experiment_uuid=config.experiment_uuid,
                )
                executor.submit(write_to_parquet, attention)  # non-blocking i/o

    slide_results = pd.DataFrame(
        slide_results,
        columns=["wsi_uuid", "prediction", "label"]
    ).assign(experiment_uuid=config.experiment_uuid, split=split)
    slide_results.to_parquet(f"{base_path}/prediction", partition_cols=["experiment_uuid"])
