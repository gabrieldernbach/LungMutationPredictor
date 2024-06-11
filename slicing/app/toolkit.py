import io
import tempfile
from uuid import uuid4

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pyvips
from PIL import Image
from skimage.color import rgb2hsv
from skimage.color import rgba2rgb
from skimage.filters import threshold_otsu
from skimage.filters import threshold_triangle
from tqdm import tqdm

Uuid = str
LocationX = int
LocationY = int
ImageBytes = bytes
TileList = [(Uuid, LocationX, LocationY, ImageBytes)]
VipsImage = pyvips.Image


def slice_wsi(cloud_path, patch_size=224, requested_mpp=0.5) -> (TileList, ImageBytes):
    with tempfile.NamedTemporaryFile() as tmp_file:
        img: VipsImage = download_image(cloud_path, tmp_file)
        img = resize(img, tmp_file.name, requested_mpp)
        img = pad(img, patch_size)
        idxs, tissue_boundary = find_tissue_tiles(tmp_file.name, img.width, patch_size)
        tiles = extract_tiles(img, patch_size, idxs)
    return tiles, tissue_boundary


def pad(img: VipsImage, patch_size: int) -> VipsImage:
    """pad image to full multiple of patch_size"""
    pad_height = (patch_size - img.height % patch_size) % patch_size
    pad_width = (patch_size - img.width % patch_size) % patch_size
    return img.embed(0, 0, img.width + pad_width, img.height + pad_height, extend="mirror")


def download_image(cloud_path: str, tmp_file) -> VipsImage:
    """get in-memory blob for fast access as well as filehandle to extract metadata"""
    with fsspec.open(cloud_path) as f:
        print(f"downloading {cloud_path}")
        buffer = io.BytesIO(f.read())
        tmp_file.write(buffer.getvalue())
        tmp_file.flush()
        return pyvips.Image.new_from_buffer(buffer.getvalue(), "")


def generate_mask(img: np.array) -> np.array:
    """A mask of foreground/background of a whole slide image's thumbnail"""
    img = rgba2rgb(img) if img.shape[-1] == 4 else img
    hue, value, saturation = np.split(rgb2hsv(img), 3, axis=-1)
    thresh = (threshold_otsu(value) + threshold_triangle(value)) / 2
    return (value > thresh).squeeze(-1)


def resize(img: VipsImage, file_name: str, requested_mpp: float) -> VipsImage:
    meta_data_handle = pyvips.Image.new_from_file(file_name)
    pixel_per_unit = meta_data_handle.get_value("xres")
    unit = get_unit(meta_data_handle)
    correction = 0.1 if unit == 'cm' else 1  # current best practice
    observed_mpp = (1000 * correction) / pixel_per_unit
    resize_factor = calculate_resize_factor(observed_mpp, requested_mpp)
    print(f"Resizing to factor: {resize_factor}")
    return img.resize(resize_factor)


def calculate_resize_factor(observed_mpp, requested_mpp) -> float:
    resize_factor = observed_mpp / requested_mpp
    if not 0.1 < resize_factor < 1.0:
        raise ValueError("Factor must be between 0.1 and 1.0")
    return resize_factor


def get_unit(img):
    fields = set(img.get_fields())
    for key in ["tiff.ResolutionUnit", "resolution-unit"]:
        if key in fields:
            return img.get_value(key)
    raise ValueError("Cannot determine resolution unit")


def find_tissue_tiles(file_name, width, patch_size):
    """get idxs of foreground tiles"""
    # thumbnail with one pixel value per patch
    n_pixel = width // patch_size - 1  # correct with 1 to get centered aggregates
    thumb = pyvips.Image.thumbnail(file_name, n_pixel).numpy()
    mask = generate_mask(thumb)
    idxs = list(zip(*np.where(mask.T)))

    fig, ax = plt.subplots()
    ax.imshow(thumb)
    ax.contour(mask)
    ax.axis("off")
    tissue_boundary = io.BytesIO()
    fig.savefig(tissue_boundary)

    return idxs, tissue_boundary


def extract_tiles(img: VipsImage, patch_size: int, idxs: [int, int]):
    def extract_tile(idx):
        x, y = idx
        tile = img.crop(
            x * patch_size,
            y * patch_size,
            patch_size,
            patch_size
        ).numpy()[..., :3]
        buffer = io.BytesIO()
        Image.fromarray(tile).save(buffer, format="JPEG")
        return str(uuid4()), x, y, buffer.getvalue()

    return [extract_tile(idx) for idx in tqdm(idxs)]
