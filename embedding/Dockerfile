FROM python:3.11.9-bullseye

RUN apt-get update && apt-get install -y vim tmux
RUN pip install numpy pandas matplotlib joblib click tqdm duckdb gcsfs pyarrow requests tenacity
WORKDIR /app
COPY app /app