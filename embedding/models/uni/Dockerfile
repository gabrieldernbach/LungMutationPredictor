#FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime  # not suitable for t4 instance
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install vim tmux -y
RUN pip install fsspec duckdb pillow pandas gcsfs pyarrow fastparquet timm jupyter

WORKDIR /app
EXPOSE 8000
RUN pip install fastapi uvicorn python-multipart tenacity # aiohttp_retry
COPY app /app
CMD ["python", "app.py"]