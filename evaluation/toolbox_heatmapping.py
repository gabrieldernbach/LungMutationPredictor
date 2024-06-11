import io

import matplotlib
import numpy as np
import pandas as pd
import pyvips as vips
from PIL import Image


def create_grid(max_x, max_y):
    """Create a DataFrame representing all possible (x, y) positions in the grid."""
    return pd.DataFrame([(x, y) for x in range(max_x) for y in range(max_y)], columns=["x", "y"])


def merge_with_tiles(grid, tiles):
    """Merge the grid with tile data, filling missing entries."""
    return grid.merge(tiles, on=["x", "y"], how='left')


def image_to_bytes(image):
    """Convert a PIL Image to JPEG format bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def fill_missing_tiles(grid):
    """Fill missing tiles with a white image."""
    white_image = Image.fromarray(np.ones((224, 224, 3), dtype="uint8") * 255)
    white_bytes = image_to_bytes(white_image)
    grid['tile'] = grid['tile'].fillna(white_bytes)
    return grid


def convert_bytes_to_vips_images(grid):
    """Convert tile data from bytes to vips images."""
    grid['tile'] = grid['tile'].apply(lambda x: vips.Image.new_from_buffer(x, options=""))
    return grid


def sort_tiles(grid):
    """Sort grid DataFrame for image assembly."""
    return grid.sort_values(by=["y", "x"])


def join_images(grid, columns):
    """Join individual tile images into a single large image."""
    return vips.Image.arrayjoin(grid['tile'].tolist(), across=columns)


def display_thumbnail(vips_image):
    """Convert a vips image to a PIL Image, resize, and display."""
    pil_image = Image.fromarray(vips_image.resize(0.02).numpy())
    return pil_image


def tiles_to_wsi(tile):
    grid = create_grid(tile["x"].max() + 1, tile["y"].max() + 1)
    grid = merge_with_tiles(grid, tile[["x", "y", "tile"]])
    grid = fill_missing_tiles(grid)
    grid = convert_bytes_to_vips_images(grid)
    grid = sort_tiles(grid)
    final_image = join_images(grid, tile['x'].max() + 1)
    return final_image


def get_heatmap_from_attention(attn, tile, patch_size: int):
    heatmap = attn.merge(tile[["tile_uuid", "x", "y"]], on="tile_uuid")
    grid = create_grid(heatmap.x.max() + 1, heatmap.y.max() + 1)
    grid = grid.merge(heatmap, on=["x", "y"], how='left')
    heatmap = grid.pivot(index="x", columns="y", values="attention").values.T

    cmap = matplotlib.colormaps["viridis"]
    cmap.set_bad(color='white')
    factor = 0.001  # very soft thresholding
    qmin, qmax = np.nanquantile(heatmap, (factor, 1 - factor))
    arr = np.clip(heatmap, qmin, qmax)
    arr = arr - np.nanmin(arr)
    arr = arr / np.nanmax(arr)
    arr = cmap(arr, bytes=True, alpha=255)
    heatmap = vips.Image.new_from_array(arr[..., :-1])
    return heatmap.resize(patch_size)


def overlay_heatmap(he_image, heatmap):
    he_gray = he_image.colourspace("b-w")
    overlayed = he_gray.composite2(heatmap, 'soft-light')
    return overlayed
