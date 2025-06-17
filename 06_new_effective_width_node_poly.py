import pandas as pd 
import numpy as np 
from loguru import logger 
from matplotlib import pyplot as plt
import rasterio
import os 
import geopandas as gpd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label
from scipy import ndimage
import cv2
from tqdm import tqdm
from glob import glob
import geopandas as gpd
import os
import rasterio
# from pyproj import Transformer

from rasterio.features import shapes
from shapely.geometry import shape
import utm

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import box
from pyproj import CRS
from rasterio.transform import xy

from pyproj import Transformer
from shapely.geometry import Polygon
import netCDF4
from rasterio.features import rasterize

parser = argparse.ArgumentParser(description='')
parser.add_argument('--idx', default=0, type=int, help='Index of merged df to process in multiples of 100 [0 to 123]')
parser.add_argument('--out_dir', default="predicted_widths_node_poly", type=str, help='Folder to save predicted widths')

def calculate_utm_zone(lon):
    """Automatically calculate UTM zone from longitude."""
    return int((lon + 180) / 6) + 1

def longlat_to_utm(x, y, zone):
    """Convert lat/lon to UTM."""
    transformer = Transformer.from_crs("epsg:4326", f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs", always_xy=True)
    x_utm, y_utm = transformer.transform(x, y)
    return np.array(x_utm), np.array(y_utm)

def extract_nodes(nc, node_ids, utm_zone):
    """Extract node attributes and UTM coordinates."""
    node_all_ids = nc.groups['nodes'].variables['node_id'][:]
    node_index = np.where(np.isin(node_all_ids, node_ids))[0]

    node_x = nc.groups['nodes'].variables['x'][:][node_index]
    node_y = nc.groups['nodes'].variables['y'][:][node_index]
    node_max_width = nc.groups['nodes'].variables['max_width'][:][node_index]
    node_ext_dist = nc.groups['nodes'].variables['ext_dist_coef'][:][node_index]
    node_length = nc.groups['nodes'].variables['node_length'][:][node_index]
    node_reachid = nc.groups['nodes'].variables['reach_id'][:][node_index]
    node_nodeid = node_all_ids[node_index]

    node_df = pd.DataFrame({
        'lon': node_x,
        'lat': node_y,
        'node_id': node_nodeid,
        'node_wmax': node_max_width,
        'node_extd': node_ext_dist,
        'node_length': node_length,
        'reach_id': node_reachid
    })

    utm_x, utm_y = longlat_to_utm(node_df['lon'].values, node_df['lat'].values, utm_zone)
    node_df['node_UTM_x'] = utm_x
    node_df['node_UTM_y'] = utm_y

    return node_df

def extract_centerlines(nc, reach_ids, utm_zone):
    """Extract centerline points and UTM coordinates."""
    cl_reach_ids = nc.groups['centerlines'].variables['reach_id'][:]
    # cl_index = np.where(np.isin(cl_reach_ids, reach_ids))[0]
    # cl_index = np.where(np.isin(cl_reach_ids, reach_ids))[1]    # 2D array, since cl_reach_ids has shape (4,N)
    cl_index_2d = np.where(cl_reach_ids==reach_id)
    cl_index = cl_index_2d[1]
    cl_index = np.unique(cl_index)
    # logger.debug(f"cl_index: {cl_index}")

    # logger.debug(f"Filtering CL based on reach_id")
    cl_x = nc.groups['centerlines'].variables['x'][:][cl_index]
    cl_y = nc.groups['centerlines'].variables['y'][:][cl_index]
    cl_id = nc.groups['centerlines'].variables['cl_id'][:][cl_index]
    # cl_node_id = nc.groups['centerlines'].variables['node_id'][:][cl_index]
    cl_node_id = nc.groups['centerlines'].variables['node_id'][:][cl_index_2d[0], cl_index_2d[1]]
    # logger.debug(f"cl_x: {cl_x}")
    # logger.debug(f"cl_y: {cl_y}")
    # logger.debug(f"cl_id: {cl_id}")
    # logger.debug(f"cl_node_id: {cl_node_id}")
    # logger.debug(f"cl_reach_ids[cl_index_2d[0], cl_index_2d[1]]: {cl_reach_ids[cl_index_2d[0], cl_index_2d[1]]}")

    # logger.debug(f"Making CL dataframe")
    cl_df = pd.DataFrame({
        'reach_id': cl_reach_ids[cl_index_2d[0], cl_index_2d[1]],
        'lon': cl_x,
        'lat': cl_y,
        'cl_id': cl_id,
        'node_id': cl_node_id
    }).dropna(subset=['lat'])

    # logger.debug(f"Adding utm of lat/lon")
    utm_x, utm_y = longlat_to_utm(cl_df['lon'].values, cl_df['lat'].values, utm_zone)
    cl_df['cl_UTM_x'] = utm_x
    cl_df['cl_UTM_y'] = utm_y

    return cl_df

def build_node_polygons(node_df, cl_df, utm_zone):
    """Build node polygons based on centerline rotation."""
    node_cls = pd.merge(node_df, cl_df, how='left', on=['node_id', 'reach_id'])

    node_polygons = []

    for node_id, group in node_cls.groupby('node_id'):
        group_sorted = group.sort_values('cl_id')
        if group_sorted.shape[0] < 2:
            continue  # skip nodes without min+max cl_id

        first = group_sorted.iloc[0]
        last = group_sorted.iloc[-1]

        dx = first['cl_UTM_x'] - last['cl_UTM_x']
        dy = first['cl_UTM_y'] - last['cl_UTM_y']
        angle = np.arctan2(dy, dx)

        offset_x = np.cos(angle - np.pi / 2) * first['node_wmax'] * first['node_extd']  # get max extent from node, perpendicular to centerline
        offset_y = np.sin(angle - np.pi / 2) * first['node_wmax'] * first['node_extd']

        p1 = (first['cl_UTM_x'] - offset_x, first['cl_UTM_y'] + offset_y)
        p2 = (last['cl_UTM_x'] - offset_x, last['cl_UTM_y'] + offset_y)
        p3 = (last['cl_UTM_x'] + offset_x, last['cl_UTM_y'] - offset_y)
        p4 = (first['cl_UTM_x'] + offset_x, first['cl_UTM_y'] - offset_y)

        poly = Polygon([p1, p2, p3, p4, p1])
        node_polygons.append({'node_id': node_id, 'reach_id': first['reach_id'], 'geometry': poly})

    node_gdf = gpd.GeoDataFrame(node_polygons, crs=f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs")
    return node_gdf


def remove_small_islands(binary_mask, min_size=500):
    labeled_mask, num_features = label(binary_mask)
    
    for i in range(1, num_features + 1):
        if np.sum(labeled_mask == i) < min_size:
            binary_mask[labeled_mask == i] = 0
    
    return binary_mask



### Get regions that intersect with SWORD reach
def get_sword_mask_intersection(raster_path, sword_gdf, tile_fp, satellite_src):
    # Load the binary mask raster
    if satellite_src == "planet-baseline":
        with rasterio.open(raster_path) as src1:
            binary_mask = src1.read(1)
        with rasterio.open(tile_fp) as src: # use input tile as source of transforms, etc
            transform = src.transform
            raster_crs = src.crs
            extent = src.bounds
    
    else:
        with rasterio.open(raster_path) as src:
            binary_mask = src.read(1)
            transform = src.transform
            raster_crs = src.crs
            extent = src.bounds


    # Ensure the line shapefile has the same CRS as the raster
    if sword_gdf.crs != raster_crs:
        sword_gdf = sword_gdf.to_crs(raster_crs)

    # Identify contiguous regions in the binary mask
    mask_labels, num_labels = ndimage.label(binary_mask)

    # Convert labeled mask regions to polygons
    shapes_list = list(shapes(mask_labels, transform=transform))  # Store generator results

    region_polygons = [shape(geom) for geom, value in shapes_list if value > 0]
    region_ids = [value for _, value in shapes_list if value > 0]  # Extract region IDs correctly

    # Create a GeoDataFrame of regions
    gdf_regions = gpd.GeoDataFrame({'region_id': region_ids, 'geometry': region_polygons}, crs=raster_crs)

    # Find intersecting regions
    intersecting_regions = gdf_regions[gdf_regions.intersects(sword_gdf.unary_union)]
    intersecting_region_ids = intersecting_regions['region_id'].tolist()
    intersecting_mask = np.isin(mask_labels, intersecting_region_ids)
    
    return transform, binary_mask, intersecting_mask

def crs_is_projected(crs: CRS) -> bool:
    """Return True if the CRS' units are linear (projected)."""
    return CRS(crs).is_projected

def choose_metric_crs(raster_crs: CRS, bounds) -> CRS:
    """
    Pick a metric CRS for accurate length measurement.

    1.  If the raster CRS is already projected → use it.
    2.  Otherwise, build a suitable UTM zone based on the raster centre.
    """
    if crs_is_projected(raster_crs):
        return raster_crs

    # derive UTM zone from the raster centre (lon,lat)
    lon_c = (bounds.left + bounds.right) / 2
    lat_c = (bounds.top + bounds.bottom) / 2
    zone_number = int((lon_c + 180) // 6) + 1
    hemi = "north" if lat_c >= 0 else "south"
    utm_crs = CRS.from_dict(
        {"proj": "utm", "zone": zone_number, "south": hemi == "south", "ellps": "WGS84"}
    )
    return utm_crs

def choose_metric_crs_nonzero_bounds(rast_crs: CRS, bounds) -> CRS:
    """
    Pick a metric CRS for accurate length measurement.

    1.  If the raster CRS is already projected → use it.
    2.  Else build an on-the-fly UTM zone centred on `bounds`.
    """
    if crs_is_projected(rast_crs):
        return rast_crs

    lon_c = (bounds[0] + bounds[2]) / 2          # minx, miny, maxx, maxy
    lat_c = (bounds[1] + bounds[3]) / 2
    zone = int((lon_c + 180) // 6) + 1
    hemi = "south" if lat_c < 0 else "north"
    return CRS.from_dict({"proj": "utm",
                          "zone": zone,
                          "south": hemi == "south",
                          "ellps": "WGS84"})


def nonzero_bounds(dataset, masked_data):
    """
    Return (minx, miny, maxx, maxy) of the raster area where values ≠ 0 / nodata.
    If *all* cells are 0 / nodata, raises a RuntimeError.
    """
    # arr = dataset.read(1, masked=True)           # masked with nodata
    mask = (masked_data != 0)                  # True where value ≠ 0
    if not mask.any():
        raise RuntimeError("Raster contains no non-zero cells.")

    rows, cols = np.nonzero(mask)
    xs, ys = xy(dataset.transform, rows, cols, offset="center")
    xs, ys = np.asarray(xs), np.asarray(ys)
    return xs.min(), ys.min(), xs.max(), ys.max()

def get_intersecting_reach_len_bounded(shp_path: Path, rast_path: Path, reach_id: int, masked_data):
    # 1. open raster & build “non-zero” footprint bounding box
    with rasterio.open(rast_path) as ras:
        rast_crs = ras.crs
        rast_bounds = ras.bounds           # in raster's native CRS
        minx, miny, maxx, maxy = nonzero_bounds(ras, masked_data)
        footprint_poly = box(minx, miny, maxx, maxy)

    # 2. read shapefile and project into raster CRS
    gdf = gpd.read_file(shp_path)
    gdf = gdf[gdf["reach_id"]==reach_id]
    if gdf.empty:
        raise RuntimeError("Shapefile contains no geometries.")
    gdf_rast = gdf.to_crs(rast_crs)

    # 3. clip by the footprint
    clipped = gdf_rast.intersection(footprint_poly)
    clipped = gpd.GeoSeries(clipped[~clipped.is_empty], crs=rast_crs)
    if clipped.empty:
        # print(0.0)
        return rast_bounds, rast_crs, 0

    # 4. re-project to a metric CRS and measure length
    metric_crs = choose_metric_crs_nonzero_bounds(rast_crs, (minx, miny, maxx, maxy))
    if metric_crs != rast_crs:
        warnings.warn(
            f"Raster CRS is geographic; using dynamic UTM "
            f"{metric_crs.to_string()} for length calculation.",
            stacklevel=2,
        )

    length_m = clipped.to_crs(metric_crs).length.sum()
    # print(round(float(length_m), 3))
    return rast_bounds, rast_crs, length_m

def get_intersecting_reach_len(shp_path: Path, rast_path: Path, reach_id: int):
    # 1. open raster & fetch its bounds/CRS
    with rasterio.open(rast_path) as ras:
        rast_bounds = ras.bounds           # in raster's native CRS
        rast_crs = ras.crs

    # 2. read vector; re-project into raster CRS so we can clip by bounds
    gdf = gpd.read_file(shp_path)
    gdf = gdf[gdf["reach_id"]==reach_id]
    if gdf.empty:
        raise RuntimeError("Shapefile contains no geometries.")

    gdf_in_rast_crs = gdf.to_crs(rast_crs)

    # 3. clip by raster bounding box
    bbox_poly = box(*rast_bounds)
    clipped = gdf_in_rast_crs.intersection(bbox_poly)

    # intersection() returns a GeoSeries; drop empties & wrap back in GeoDataFrame
    clipped = gpd.GeoSeries(clipped[~clipped.is_empty], crs=rast_crs)
    if clipped.empty:
        # print(0.0)
        return rast_bounds, rast_crs, 0

    # 4. choose a metric CRS (either raster's projected CRS, or on-the-fly UTM)
    metric_crs = choose_metric_crs(rast_crs, rast_bounds)
    if metric_crs != rast_crs:
        warnings.warn(
            f"Raster CRS is geographic; using dynamical UTM {metric_crs.to_string()} "
            "for length calculation.",
            stacklevel=2,
        )

    clipped_m = clipped.to_crs(metric_crs)

    # 5. compute & print length in metres
    total_m = clipped_m.length.sum()
    # print(round(float(total_m), 3))
    return rast_bounds, rast_crs, total_m


if __name__=="__main__":
    args = parser.parse_args()
    logger.debug(f"args: {args}")

    INTERVAL = 100
    RESOLUTION = 3      # 3m/pixel for planet
    planet_dir = "/project/pi_cjgleason_umass_edu/swot_planet/unit/" 
    ref_df_fp = "/project/pi_cjgleason_umass_edu/swot_planet/yukon_imgs.csv"
    ref_df = pd.read_csv(ref_df_fp)

    sword_mapping_fp = "all_sword_mapping_v16.csv" # SWORDv16
    sword_mapping = pd.read_csv(sword_mapping_fp)

    all_paths = []
    for fp in glob(os.path.join("tiled_planet_predicted_merged", "*.tif")):
        all_paths.append(fp)
    all_paths = sorted(all_paths)
    logger.debug(f"Found {len(all_paths)} rasters to process")

    start_idx = args.idx*INTERVAL
    end_idx = (args.idx+1)*INTERVAL
    logger.debug(f"start_idx: {start_idx}. end_idx: {end_idx}")

    filtered_paths = all_paths[start_idx:end_idx]
    logger.debug(f"Processing {len(filtered_paths)} rasters")

    all_predicted_data = [] # reach_id, planet_fp, pred_fp, reach_len_full, reach_len_raster_bound, reach_len_pred_bound, water_area, sword_width
    for pred_raster_fp in tqdm(filtered_paths):
        reach_id = int(float(pred_raster_fp.split("/")[-1].split("--")[0]))
        planet_rel_path = pred_raster_fp.split("/")[-1].replace("--","/")
        planet_fp = os.path.join(planet_dir, planet_rel_path)
        assert os.path.exists(planet_fp)

        # Find corresponding SWORD fp
        filtered_sword = sword_mapping[sword_mapping["reach_id"]==reach_id]
        assert len(filtered_sword)==1
        sword_fp = filtered_sword.iloc[0]["sword_fp"]
        gdf = gpd.read_file(sword_fp)
        sword_gdf = gdf[gdf["reach_id"]==reach_id]  # filter relevant reach_id

        watermask_raster_transform, binary_mask, intersecting_mask = get_sword_mask_intersection(
            raster_path=pred_raster_fp, sword_gdf=sword_gdf, tile_fp=pred_raster_fp, satellite_src="planet"
        )
        num_water_px = np.sum(intersecting_mask.astype(int))
        perc_water_px = (num_water_px)/(intersecting_mask.shape[0]*intersecting_mask.shape[1])
        water_area = num_water_px*(RESOLUTION*RESOLUTION)

        # Compute correct reach length that is within the water mask
        if num_water_px>0:
            raster_bounds, raster_crs, reach_len_pred_bound = get_intersecting_reach_len_bounded(
                shp_path=sword_fp, rast_path=pred_raster_fp, reach_id=reach_id, masked_data=intersecting_mask
            )
        else:
            reach_len_pred_bound = 0
        raster_bounds, raster_crs, reach_len_raster_bound = get_intersecting_reach_len(
            shp_path=sword_fp, rast_path=pred_raster_fp, reach_id=reach_id
        )

        # Only take water pixels that are within the node polygons
        continent = sword_fp.split('/')[-2].lower()
        nc_files = glob(os.path.join("/".join(sword_fp.split('/')[:-3]), "netcdf", f"*{continent}*"))
        assert len(nc_files)==1
        sword_nc_file = nc_files[0]
        # logger.debug(f"sword_nc_file: {sword_nc_file}")

        assert os.path.exists(sword_nc_file)
        sword_nc = netCDF4.Dataset(sword_nc_file, mode='r')
        reach_ids = [reach_id]
        reach_all_ids = sword_nc.groups['reaches'].variables['reach_id'][:]
        reach_idx = np.where(np.isin(reach_all_ids, reach_ids))[0]
        reach_x = sword_nc.groups['reaches'].variables['x'][:][reach_idx]
        utm_zone = calculate_utm_zone(reach_x[0])
        # logger.debug(f"utm_zone: {utm_zone}")

        all_reach_ids_node = sword_nc["nodes"]["reach_id"][:]
        node_ids_idx = np.where(np.isin(all_reach_ids_node, reach_ids))[0]
        node_ids = sword_nc.groups['nodes'].variables['node_id'][:][node_ids_idx]
        # logger.debug(f"node_ids: {node_ids}")

        
        node_df = extract_nodes(sword_nc, node_ids, utm_zone)
        cl_df = extract_centerlines(sword_nc, reach_ids, utm_zone)
        node_gdf = build_node_polygons(node_df, cl_df, utm_zone)
        node_gdf['geometry'] = node_gdf.buffer(200)     # as per Taylor's script, buffer to remove gaps


        # Plot node polygons overlaid
        sword_gdf_raster = sword_gdf.to_crs(raster_crs)
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        ax1.imshow(intersecting_mask, extent=[raster_bounds.left, raster_bounds.right,raster_bounds.bottom, raster_bounds.top])
        ax1.set_xlim(raster_bounds.left, raster_bounds.right)
        ax1.set_ylim(raster_bounds.bottom, raster_bounds.top)
        # plt.imshow()
        node_gdf.plot(ax=ax1, color="blue")
        sword_gdf_raster.plot(ax=ax1, color="red")
        plt.savefig(os.path.join(args.out_dir, "imgs", f"{pred_raster_fp.split('/')[-1].replace('.tif','_node_poly.png')}"), bbox_inches="tight")
        plt.close()


        node_shapes = [(geom, 1) for geom in node_gdf.geometry]  # NOT node_gdf.geometry()
        node_mask = rasterize(
            node_shapes,
            out_shape=intersecting_mask.shape,
            transform=watermask_raster_transform,
            fill=0,
            dtype=np.uint8
        )
        updated_mask = (node_mask&intersecting_mask)
        updated_num_water_px = np.sum(updated_mask.astype(int))
        updated_perc_water_px = (updated_num_water_px)/(updated_mask.shape[0]*updated_mask.shape[1])
        updated_water_area = updated_num_water_px*(RESOLUTION*RESOLUTION)

        
        reach_len_full = sword_gdf.iloc[0]["reach_len"]
        sword_width = sword_gdf.iloc[0]["width"]
        filtered_ref_df = ref_df[ref_df["planet_dir"]==planet_fp]
        assert len(filtered_ref_df)==1
        cloud_cover = filtered_ref_df["planet_cloud_cover"].iloc[0]

        all_predicted_data.append([
            reach_id, planet_fp, cloud_cover, pred_raster_fp, reach_len_full, reach_len_raster_bound, 
            reach_len_pred_bound, num_water_px, perc_water_px, water_area, sword_width,
            updated_num_water_px, updated_perc_water_px, updated_water_area,
        ])

        df = pd.DataFrame(all_predicted_data, columns=[
            "reach_id", "planet_fp", "cloud_cover", "pred_raster_fp", "reach_len_full", "reach_len_raster_bound", 
            "reach_len_pred_bound", "num_water_px", "perc_water_px", "water_area", "sword_width",
            "updated_num_water_px", "updated_perc_water_px", "updated_water_area",
        ])
        df.to_csv(os.path.join(args.out_dir, f"{start_idx:04d}_{end_idx:04d}_pred_widths.csv"), index=False)
        logger.debug(f"Saved results to {start_idx:04d}_{end_idx:04d}_pred_widths.csv")


        sword_gdf_raster = sword_gdf.to_crs(raster_crs)
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        ax1.imshow(intersecting_mask, extent=[raster_bounds.left, raster_bounds.right,raster_bounds.bottom, raster_bounds.top])
        ax1.set_xlim(raster_bounds.left, raster_bounds.right)
        ax1.set_ylim(raster_bounds.bottom, raster_bounds.top)
        # plt.imshow()
        sword_gdf_raster.plot(ax=ax1)
        plt.savefig(os.path.join(args.out_dir, "imgs", f"{pred_raster_fp.split('/')[-1].replace('.tif','.png')}"), bbox_inches="tight")
        plt.close()


        fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        ax1.imshow(updated_mask, extent=[raster_bounds.left, raster_bounds.right,raster_bounds.bottom, raster_bounds.top])
        ax1.set_xlim(raster_bounds.left, raster_bounds.right)
        ax1.set_ylim(raster_bounds.bottom, raster_bounds.top)
        # plt.imshow()
        sword_gdf_raster.plot(ax=ax1)
        plt.savefig(os.path.join(args.out_dir, "imgs", f"{pred_raster_fp.split('/')[-1].replace('.tif','_nodepoly_updated.png')}"), bbox_inches="tight")
        plt.close()

        sword_nc.close()