# This code is used to merge planet predictions into one big tiff file for easier processing and computation of effective widths

import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from glob import glob
from loguru import logger
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='')
parser.add_argument('--idx', default=0, type=int, help='Index of merged df to process in multiples of 100 [0 to 123]')
parser.add_argument('--out_dir', default="tiled_planet_predicted_merged", type=str, help='Folder to save tiled planet images')



def merge_rasters(in_paths, out_path, resampling=Resampling.nearest) -> None:
    """
    Merge a collection of rasters and save to `out_path`.

    Parameters
    ----------
    in_paths : Sequence[str | Path]
        Ordered list/tuple of raster file paths to merge.
    out_path : str | Path
        Output raster path (GeoTIFF or any rasterio-supported format).
    resampling : rasterio.enums.Resampling, optional
        Resampling method used when inputs differ in resolution. Default is nearest‐neighbour.

    Raises
    ------
    FileNotFoundError
        If any path in `in_paths` does not exist.
    ValueError
        If `in_paths` is empty.
    """
    in_paths = [Path(p) for p in in_paths]
    if not in_paths:
        raise ValueError("`in_paths` is empty.")

    # Validate inputs
    for p in in_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    # Open datasets
    datasets = [rasterio.open(p) for p in in_paths]

    # Merge (returns a single numpy array + updated transform)
    mosaic, transform = merge(datasets, resampling=resampling)

    # Copy profile from first file and update for output
    profile = datasets[0].profile
    profile.update(
        driver="GTiff",          # change if you need another format
        height=mosaic.shape[1],  # (bands, rows, cols) → rows index 1
        width=mosaic.shape[2],   # cols index 2
        transform=transform,
        count=mosaic.shape[0],   # number of bands
        compress="lzw"           # good default compression
    )

    # Write merged raster
    out_path = Path(out_path)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)

    # Clean up
    for ds in datasets:
        ds.close()

    # print(f"Merged raster written to {out_path.resolve()}")


if __name__=="__main__":
    args = parser.parse_args()
    logger.debug(f"args: {args}")
    INTERVAL = 100


    start_idx = args.idx*INTERVAL
    end_idx = (args.idx+1)*INTERVAL
    logger.debug(f"start_idx: {start_idx}. end_idx: {end_idx}")

    in_dir = "tiled_planet_predicted"
    dir_names = sorted(os.listdir(in_dir))
    logger.debug(f"TOTAL dir_names: {len(dir_names)}")

    filtered_dir_names = dir_names[start_idx:end_idx]
    logger.debug(f"Processing {len(filtered_dir_names)} filtered_dir_names")
    logger.debug(f"Merge planet predictions")
    
    for dir_name in tqdm(filtered_dir_names):
        in_tifs = glob(os.path.join(in_dir, dir_name, "*.tif"))
        # print(in_tifs, len(in_tifs))
        out_path = os.path.join(args.out_dir, dir_name+".tif")
        merge_rasters(in_tifs, out_path, resampling=Resampling.nearest)
        logger.debug(f"out_path: {out_path}. len(in_tifs): {len(in_tifs)}")