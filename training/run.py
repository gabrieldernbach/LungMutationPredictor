import logging
import time
import uuid

import fsspec
import pandas as pd
import torch

from data import get_dataloaders, get_holdout
from fit_predict import fit, predict
from gen_experiment import Config
from model import get_learner

base_path = "gs://bucket_name/iteration_n/data"



class FileLock:
    def __init__(self, lock_path, max_retries=20, retry_interval=5, timeout=120):
        self.lock_path = lock_path
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.timeout = timeout
        self.gfs = fsspec.filesystem('gs')
        self.lock_acquired = False
        self.uuid = str(uuid.uuid4())

    def __enter__(self):
        start_time = time.time()
        retries = 0

        while not self.lock_acquired:
            if not self.gfs.exists(self.lock_path):
                try:
                    # Attempt to create the lock file with UUID atomically
                    with self.gfs.open(self.lock_path, 'w') as lock_file:
                        lock_file.write(self.uuid)
                    self.lock_acquired = True
                    break
                except Exception as e:
                    print(f"Failed to acquire lock: {e}")
            else:
                print(f"Lock exists. Verifying ownership...")

                try:
                    with self.gfs.open(self.lock_path, 'r') as lock_file:
                        existing_uuid = lock_file.read().strip()
                    if existing_uuid == self.uuid:
                        self.lock_acquired = True
                        break
                    else:
                        print(f"Lock owned by another process. Retrying in {self.retry_interval} seconds...")
                except Exception as e:
                    print(f"Failed to read lock file: {e}")

                time.sleep(self.retry_interval)
                retries += 1
                if retries >= self.max_retries or (time.time() - start_time) > self.timeout:
                    raise TimeoutError("Failed to acquire lock within the specified timeout period")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.lock_acquired:
            retries = 0
            while retries < self.max_retries:
                try:
                    with self.gfs.open(self.lock_path, 'r') as lock_file:
                        existing_uuid = lock_file.read().strip()
                    if existing_uuid == self.uuid:
                        self.gfs.rm(self.lock_path)
                        self.lock_acquired = False
                        break
                    else:
                        logging.error(f"Lock owned by another process. Manual intervention required.")
                        break
                except Exception as e:
                    retries += 1
                    logging.error(f"Failed to release lock: {e}. Retrying ({retries}/{self.max_retries})...")
                    time.sleep(self.retry_interval)
            if self.lock_acquired:
                logging.error(f"Failed to release lock after {self.max_retries} attempts. Manual intervention required.")


def get_config():
    file_path = f"{base_path}/experiments.parquet"
    lock_path = file_path + ".lock"
    with FileLock(lock_path):
        df = pd.read_parquet(file_path)
        if (df.status == "pending").any():
            entry = df.query("status=='pending'").sample(1).iloc[0]
            df.loc[df.experiment_uuid == entry.experiment_uuid, "status"] = "processing"
            df.to_parquet(file_path)
            return Config(**entry.to_dict())
        else:
            return None


def report_success(cfg):
    file_path = f"{base_path}/experiments.parquet"
    lock_path = file_path + ".lock"
    with FileLock(lock_path):
        df = pd.read_parquet(file_path)
        df.loc[df.experiment_uuid == cfg.experiment_uuid, "status"] = "success"
        df.to_parquet(file_path)


def run(cfg):
    learner = get_learner()
    dataloaders = get_dataloaders(cfg)
    holdout_loader = get_holdout(cfg)

    model = fit(learner, dataloaders)
    with fsspec.open(f"{base_path}/checkpoint/{cfg.experiment_uuid}", "wb") as fhandle:
        torch.save(model.state_dict(), fhandle)
    predict(model, dataloaders.train_all, cfg, split="train")
    predict(model, dataloaders.dev, cfg, split="dev")
    predict(model, dataloaders.test, cfg, split="test")
    predict(model, holdout_loader, cfg, split="holdout")

def worker():
    while True:
        cfg = get_config()
        print(f"starting with config {cfg}")
        if cfg is None:
            print("No more jobs to process, worker exit")
            return
        run(cfg)
        report_success(cfg)


if __name__ == "__main__":
    worker()
