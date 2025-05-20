import os
import torch
import rasterio
import cv2
import numpy as np
import torch.nn as nn
from models.get_model import get_model
import numpy as np
import pandas as pd
from progress.bar import Bar as Bar
from utils_dir import Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils_dir.dense_losses import get_dense_tasks_losses, get_task_loss, compute_miou, compute_iou, depth_error, normal_error, compute_mask_metrics
# from torch.autograd import Variable
# from mgda.min_norm_solvers import MinNormSolver, gradient_normalizers

from dataset.planet_segmentation import PlanetSegmentation
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

from loguru import logger
from datetime import datetime
from tqdm import tqdm
import argparse
from skimage.transform import rescale, resize

EPS=1e-7




parser = argparse.ArgumentParser(description='')
parser.add_argument('--to_save_imgs', default=1, type=int, help='Set to 1 to save image outputs')
# parser.add_argument('--is_downsample', default=0, type=int, help='Set to 1 to downsample then upsample input images (to simulate lower res)')
parser.add_argument('--thresh', default=0.3, type=float, help='Set to optimal thresh value 0-1')

parser.add_argument('--tasks', default=["water_mask"], nargs='+', help='Task(s) to be trained')
parser.add_argument('--ckpt_path', default="/work/pi_smaji_umass_edu/rdaroya/planet-benchmark/results/planet-water-best-perf/20250411-100516--fpn--linear--resnet50--no_head--mtl_baselines_vanilla_uniform_model_best.pth.tar", type=str, help='specify location of checkpoint')
parser.add_argument('--pretrained', default=False, type=int, help='using pretrained weight from ImageNet')


parser.add_argument('--out', default='tiled_planet_predicted', help='Directory to output the result')



def save_imgs(out_img_dir, input_fps, test_labels, pred_water_mask, to_save_rgb=True):
    test_labels_np = test_labels.detach().cpu().numpy()
    pred_mask_np = pred_water_mask.detach().cpu().numpy()
    for idx in range(len(input_fps)):
        input_fp = input_fps[idx]
        # test_label = test_labels_np[idx]    # 0 and 1 values
        pred_mask = pred_mask_np[idx]       # 0 and 1 values
        
        out_name = input_fp.split("/")[-1]
        out_fp_tif = os.path.join(out_img_dir, out_name)
        # logger.debug(f"Saving output to {out_fp_tif}")

        input_dataset = rasterio.open(input_fp)
        # Write prediction to TIFF
        kwargs = input_dataset.meta
        kwargs.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw')
        with rasterio.open(out_fp_tif, 'w', **kwargs) as dst:
            dst.write_band(1, pred_mask.astype(rasterio.float32))
        # Write prediction to PNG
        out_fp_png = out_fp_tif.replace(".tif", ".png")
        cv2.imwrite(out_fp_png, pred_mask*255)

        # RGB image
        if to_save_rgb:
            img = input_dataset.read()
            img = np.transpose(img, (1,2,0))
            img = (img-np.min(img))/(np.max(img)-np.min(img))
            out_name_rgb = out_fp_tif.replace(".tif", "--rgb.png")
            cv2.imwrite(out_name_rgb, img[:,:,:3]*255)





opt = parser.parse_args()
logger.debug(f"opt: {opt}")

if not os.path.isdir(opt.out):
    mkdir_p(opt.out)

to_save_imgs = (opt.to_save_imgs!=0)
logger.debug(f"to_save_imgs: {to_save_imgs}")

tasks = opt.tasks
num_inp_feats = 4   # number of channels in input
tasks_outputs_tmp = {
    "water_mask": 1,
}
tasks_outputs = {t: tasks_outputs_tmp[t] for t in tasks}
logger.debug(f"opt: {opt.__dict__}")


logger.debug(f"Loading weights from {opt.ckpt_path}")
checkpoint = torch.load(opt.ckpt_path, weights_only=False)
ckpt_fn = opt.ckpt_path.split("/")[-1].replace(".pth.tar", "")
logger.debug(f"ckpt_fn: {ckpt_fn}")
ckpt_opt = checkpoint["opt"]
ckpt_lr = ckpt_opt.lr
logger.debug(f"ckpt_lr: {ckpt_lr}")
model = get_model(ckpt_opt, tasks_outputs=tasks_outputs, num_inp_feats=num_inp_feats, pretrained=(ckpt_opt.pretrained==1))

new_ckpt = {k.split("module.")[-1]:v for k,v in checkpoint["state_dict"].items()}
checkpoint["state_dict"] = new_ckpt
tmp = model.load_state_dict(checkpoint["state_dict"], strict=True)

logger.debug(f"After loading ckpt: {tmp}")
logger.debug(f"Checkpoint epoch: {checkpoint['epoch']}. best_perf: {checkpoint['best_performance']}")
model.cuda()
model.eval()

assert opt.thresh is not None
optim_threshes = {"water_mask": opt.thresh}
logger.debug(f"Using the following thresholds: {optim_threshes}")


test_dataset1 = PlanetSegmentation(resize_size=ckpt_opt.resize_size, return_fp=True, is_downsample=False)
logger.debug(f"Using batch size 1 for test loader")
test_sampler = None
test_loader = torch.utils.data.DataLoader(
    test_dataset1, batch_size=1, shuffle=False,
    num_workers=1, pin_memory=True, sampler=test_sampler, drop_last=False)
test_batch = len(test_loader)
test_dataset = iter(test_loader)


logger.debug(f"Predicting on {test_batch} test batches")

rgbs = []
with torch.no_grad():
    for k in tqdm(range(test_batch)):
        test_data, test_labels, input_fp = next(test_dataset)
        
        gt_water_mask = test_labels["water_mask"] 
        gt_water_mask = torch.squeeze(gt_water_mask, 1).cuda()

        test_data = test_data.cuda()
        test_pred, feat = model(test_data, feat=True)

        pred = test_pred["water_mask"]
        thresh_pred = torch.where(pred > optim_threshes["water_mask"], 1., 0.)
        pred_water_mask = thresh_pred
        
        pred_water_mask = torch.squeeze(pred_water_mask, 1)

        folder_name = input_fp[0].split("/")[-2]
        out_img_dir = os.path.join(opt.out, folder_name)
        if to_save_imgs:
            # logger.debug(f"Creating output folder for images: {out_img_dir}")
            mkdir_p(out_img_dir)
        if to_save_imgs:
            save_imgs(out_img_dir, input_fp, test_labels["water_mask"], pred_water_mask)

