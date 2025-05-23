import pandas as pd 
import rasterio
import numpy as np 
import os
import matplotlib.pyplot as plt
import cv2
import glob
import geopandas as gpd
from shapely.geometry import box

from tqdm import tqdm
import pyproj

import json
from loguru import logger

from rasterio import windows

import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--idx', default=0, type=int, help='Index of merged df to process in multiples of 100 [0 to 123]')
parser.add_argument('--out_dir', default="tiled_planet", type=str, help='Folder to save tiled planet images')


# NOTE: x is lon, y is lat
def get_tile_idxs(sentinel_data, tile_size=500):
    _, H, W = sentinel_data.shape
    tiles_index_pairs = []
    for row_start in range(0,max(0,H - tile_size) + tile_size, tile_size):
        for col_start in range(0, max(0,W - tile_size) + tile_size, tile_size):
            row_end = row_start + tile_size
            col_end = col_start + tile_size
            tiles_index_pairs.append(((row_start, row_end), (col_start, col_end)))
    return tiles_index_pairs

def mkdir_p(path):
    '''make dir if not exist'''
    os.makedirs(path, exist_ok=True)


def get_tiles(ds, tile_indices, mid_idx=None):
    nols, nrows = ds.meta['width'], ds.meta['height']


    crs_src = pyproj.CRS(str(ds.crs))
    crs_dest = pyproj.CRS("EPSG:4326")  #latlon CRS
    to_latlon_transform = pyproj.Transformer.from_crs(crs_src, crs_dest)

    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for i, (row_idxs, col_idxs) in enumerate(tile_indices):
        is_mid = False
        # if i==mid_idx:
        #     is_mid=True
        col_off = col_idxs[0]
        row_off = row_idxs[0]
        width = col_idxs[1]-col_idxs[0]
        height = row_idxs[1]-row_idxs[0]
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)

        mid_idx_row = (row_idxs[1]+row_idxs[0])//2
        mid_idx_col = (col_idxs[1]+col_idxs[0])//2
        samp_mid_row, samp_mid_col = ds.xy(mid_idx_row, mid_idx_col)
        samp_mid_lat, samp_mid_lon = to_latlon_transform.transform(samp_mid_row, samp_mid_col) # transform lat/lon coordinates to CRS of image


        yield window, transform, is_mid, samp_mid_lat, samp_mid_lon
        

if __name__=="__main__":
    args = parser.parse_args()
    logger.debug(f"args: {args}")

    TILE_SIZE = 500 # pixels
    INTERVAL = 100

    # imgs_dir = "/project/pi_cjgleason_umass_edu/swot_planet/unit/"
    ref_df_fp = "/project/pi_cjgleason_umass_edu/swot_planet/yukon_imgs.csv"
    ref_df = pd.read_csv(ref_df_fp)
    logger.debug(f"Processing ref_df: {ref_df.shape}")

    start_idx = args.idx*INTERVAL
    end_idx = (args.idx+1)*INTERVAL
    logger.debug(f"start_idx: {start_idx}. end_idx: {end_idx}")

    all_tiles_record = []   # reach_id, planet_id, sentinel_fp, tile_fp, tile_width, tile_height, is_mid_node, mid_lat, mid_lon, start_row_idx, end_row_idx, start_col_idx, end_col_idx
    for idx in tqdm(range(start_idx, end_idx)):
        try:
            row = ref_df.iloc[idx]
        except:
            continue

        folder_path = "--".join(row["planet_dir"].split("/")[-4:])
        folder_path = folder_path.replace(".tif","")    # path for saving 500x500 tiles for segmentation
        logger.debug(f"Out folder: {folder_path}")
        out_path = os.path.join(args.out_dir, folder_path)
        if not os.path.exists(out_path):
            logger.debug(f"Creating out_path: {out_path}")
            os.makedirs(out_path, exist_ok=True)
        
        planet_fp = row["planet_dir"]
        reach_id = int(row["reach_id"])
        try:
            planet_data = rasterio.open(planet_fp).read()
        except Exception as e:
            logger.error(f"{e}: {planet_fp}")
            continue

        # mid_row, mid_col = get_mid_ref_idxs(sentinel_img_dataset, nodes_df, reach_id, intersecting_sword_gpd)
        tiles_index_pairs = get_tile_idxs(planet_data, tile_size=TILE_SIZE)


        output_filename = 'tile_{}_{}-{}-{}.tif'    # naming of tiles
        with rasterio.open(planet_fp) as dataset:

            meta = dataset.meta.copy()

            for window, transform, is_mid, mid_lat, mid_lon in get_tiles(dataset, tiles_index_pairs):
                # print(window)
                meta['transform'] = transform
                meta['width'], meta['height'] = window.width, window.height
                outpath = os.path.join(out_path,output_filename.format(int(window.row_off), int(window.row_off)+int(window.height), int(window.col_off), int(window.col_off)+int(window.width)))
                
                with rasterio.open(outpath, 'w', **meta) as outds:
                    outds.write(dataset.read(window=window))
                    
                record_data = [
                    reach_id, planet_fp, outpath, window.width, window.height, mid_lat, mid_lon, 
                    int(window.row_off), int(window.row_off)+int(window.height),
                    int(window.col_off), int(window.col_off)+int(window.width),
                ]
                all_tiles_record.append(record_data)
                # TODO: need to check is_reach_intersects for all tiles later
    df = pd.DataFrame(all_tiles_record, columns=[
        "reach_id", "planet_fp", "tile_fp", "tile_width", "tile_height", "mid_lat", "mid_lon", 
        "start_row_idx", "end_row_idx", 
        "start_col_idx", "end_col_idx",
    ])
    df.to_csv(os.path.join(args.out_dir, f"{start_idx:04d}_{end_idx:04d}_tiles_data.csv"), index=False)
    logger.debug(f"Saved results to {start_idx:04d}_{end_idx:04d}_tiles_data.csv")