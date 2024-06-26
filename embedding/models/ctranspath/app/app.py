import asyncio
import os
import io
from collections import deque

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, Response, Request

from model import load_model_and_preprocessor

app = FastAPI()
model, preprocessor = load_model_and_preprocessor()

# max number of elements in batch collection
batch_size = os.getenv("batch_size", 32)
# max time for collecting batch elements
batch_timeout = os.getenv("batch_timeout", 1.0)

# Shared state using deque for efficient append and pop operations
queue = deque()
queue_lock = asyncio.Lock()  # Lock to manage access to the current_batch


def serialize(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


async def process_images(images):
    try:
        image_tensors = torch.stack([
            preprocessor(Image.open(io.BytesIO(img))) for img in images
        ])
        with torch.no_grad():
            embeddings = model(image_tensors).numpy()
            embeddings = [serialize(arr) for arr in embeddings]
            return embeddings
    except Exception as e:
        raise RuntimeError(f"Error processing images: {str(e)}")


async def process_batch(batch):
    images, futures = zip(*batch)
    try:
        embeddings = await process_images(list(images))
        for future, embedding in zip(futures, embeddings):
            future.set_result(embedding)
    except Exception as e:
        for future in futures:
            future.set_exception(e)


async def batch_collector():
    while True:
        await asyncio.sleep(batch_timeout)
        image_batch = None
        async with queue_lock:
            if queue:
                image_batch = list(queue)
                queue.clear()
        if image_batch:
            asyncio.create_task(process_batch(image_batch))


@app.on_event("startup")
async def start_batch_collector():
    asyncio.create_task(batch_collector())


@app.post("/embed_image/", response_class=Response)
async def embed_image(request: Request):
    img_bytes = await request.body()
    result = asyncio.get_event_loop().create_future()
    async with queue_lock:
        queue.append((img_bytes, result))
        if len(queue) >= batch_size:
            batch_to_process = list(queue)
            queue.clear()
            asyncio.create_task(process_batch(batch_to_process))
    try:
        embedding = await result
        return Response(content=embedding, media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
