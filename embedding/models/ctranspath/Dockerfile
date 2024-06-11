FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN apt-get update; apt-get install curl vim -y

RUN pip install gdown numpy pillow
RUN gdown https://drive.google.com/file/d/1JV7aj9rKqGedXY1TdDfi3dP07022hcgZ/view?usp=sharing --fuzzy
RUN pip install timm-0.5.4.tar && rm timm-0.5.4.tar

WORKDIR /app
RUN gdown https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX/view?usp=sharing --fuzzy -O /app/ctranspath.pth

EXPOSE 8000
RUN pip install fastapi uvicorn python-multipart tenacity aiohttp_retry
COPY app /app
CMD ["python", "app.py"]