import io

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, Response, Request

from model import load_model_and_preprocessor

app = FastAPI()
model, preprocessor = load_model_and_preprocessor()


def serialize(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def process_image(image_bytes):
    try:
        image_tensors = preprocessor(Image.open(image_bytes)).unsqueeze(0)
        with torch.no_grad():
            embedding = model(image_tensors).numpy()[0]
            embedding = serialize(embedding)
            return embedding
    except Exception as e:
        raise RuntimeError(f"Error processing images: {str(e)}")


@app.post("/embed_image/", response_class=Response)
async def embed_image(request: Request):
    try:
        img_bytes = request.body()
        embedding = process_image(img_bytes)
        return Response(content=embedding, media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
