FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y vim tmux htop libvips
RUN pip install pandas tqdm duckdb gcsfs pyarrow scikit-learn jupyter pyvips matplotlib
WORKDIR /app
COPY . /app