import io

import fsspec
import timm
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_model():
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True
    )
    with fsspec.open("gs://bucket_name/iteration_n/models/uni_ckpt.bin", "rb") as f:
        buffer = io.BytesIO(f.read())
    model.load_state_dict(torch.load(buffer, map_location="cpu"), strict=True)
    model.eval()
    return model


def get_transform():
    return transforms.Compose(
        [
            # transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ]
    )


def load_model_and_preprocessor():
    return get_model(), get_transform()

