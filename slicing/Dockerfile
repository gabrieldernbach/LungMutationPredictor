FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt-get update --fix-missing && apt-get install -y vim git gcc
RUN apt-get update; apt-get install openslide-tools libtiff-dev libvips -y
RUN pip install numpy scipy pandas \
        scikit-learn matplotlib \
        pillow scikit-image joblib \
        click tqdm jupyter einops \
        openslide-python \
        pyvips duckdb gcsfs pyarrow
WORKDIR /app
COPY app /app