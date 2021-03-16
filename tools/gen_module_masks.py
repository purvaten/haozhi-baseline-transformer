r"""
python tools/gen_module_masks.py \
--split train \
--input_size 1 \
--pred_size 5 \
--start 0 \
--end 20000 \
--mask_thresh 1.0

python tools/gen_module_masks.py \
--split test \
--input_size 1 \
--pred_size 10 \
--start 0 \
--end 20000 \
--mask_thresh 1.0
"""
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import numpy as np
import argparse
import random
import hickle
import phyre
import torch
import cv2
import pdb
import sys
import os

sys.path.append('/srv/flash1/purva/Projects/physics/haozhi-baseline-transformer/')
from neuralphys.utils.bbox import xyxy2xywh
from neuralphys.utils.modules import module_mask, check_overlap, singular_module


def arg_parse():
    # only the most general argument is passed here
    # task-specific parameters should be passed by config file
    parser = argparse.ArgumentParser(description='Module generation parameters')
    parser.add_argument('--split', required=True, type=str)
    parser.add_argument('--input_size', required=True, type=int)
    parser.add_argument('--pred_size', required=True, type=int)
    parser.add_argument('--mask_thresh', required=True, type=float)
    parser.add_argument('--start', required=True, type=int)
    parser.add_argument('--end', required=True, type=int)
    return parser.parse_args()


# ******************************************* #
# CONFIG
MASK_SIZE = 21
MASK_LOSS_WEIGHT = 0.003
NUM_OBJS = 6
SINGULAR_MODULES = ['ball', 'bar', 'jar']
DUAL_MODULES = ['ball-ball', 'ball-bar', 'ball-jar', 'bar-bar', 'bar-jar', 'jar-jar']
number_to_obj_mapping = {1: 'ball', 2: 'bar', 3: 'jar'}

# DATA PATH
DATA = '/srv/flash1/purva/Projects/physics/data/PHYRE_1fps_p20n80_new/'

# IMAGE HEIGHT, WIDTH
input_width, input_height = 128, 128
# ******************************************* #


def image_colors_to_onehot(indices):
    onehot = torch.nn.functional.embedding(
        indices, torch.eye(phyre.NUM_COLORS, device=indices.device))
    onehot = onehot.permute(2, 0, 1).contiguous()
    return onehot


def parse_image(video_name, vid_idx, img_idx, seq_size):
    images = hickle.load(video_name.replace('images', 'full').replace('.npy', '_image.hkl'))
    data = np.array([phyre.observations_to_float_rgb(img.astype(int)) for img in images], dtype=np.float).transpose((0, 3, 1, 2))
    data = data[img_idx:img_idx + seq_size]
    return data


def parse_label(anno_name, vid_idx, img_idx, input_size, pred_size):
    seq_size = input_size + pred_size
    boxes = hickle.load(anno_name)[img_idx:img_idx + seq_size, :, 1:]
    gt_masks = np.zeros((pred_size, boxes.shape[1], MASK_SIZE, MASK_SIZE))
    if MASK_LOSS_WEIGHT > 0:
        gt_masks = hickle.load(anno_name.replace('boxes.', 'masks.'))
        gt_masks = gt_masks[img_idx:img_idx + seq_size].astype(np.float32)
        gt_masks = gt_masks[input_size:]

    labels = torch.zeros(1)

    # shape-related details for (static + dynamic) objects
    shapes = hickle.load(anno_name.replace('boxes.', 'shapes.'))

    # information about whether objects are static or dynamic
    # dynamics = hickle.load(anno_name.replace('boxes.', 'dynamics.'))[img_idx:img_idx + seq_size][0]
    num_objs = shapes.shape[0]
    dynamics = np.ones(num_objs)

    return boxes, gt_masks, labels, dynamics, shapes


def saveinfo(video_info, anno_list, video_list, start_idx, end_idx, input_size, pred_size, mask_thresh):
    seq_size = input_size + pred_size
    fname_gtindicator = DATA+'/thresh/'+str(mask_thresh)+'/'+str(start_idx)+'_'+str(end_idx)+'_within_'+split+'_'+str(input_size)+'_'+str(pred_size)+'_fold_0_gtindicatorinfo.npy'
    fname_obj = DATA+'/thresh/'+str(mask_thresh)+'/'+str(start_idx)+'_'+str(end_idx)+'_within_'+split+'_'+str(input_size)+'_'+str(pred_size)+'_fold_0_objinfo.npy'
    objectdata = np.empty((0, NUM_OBJS))
    gtindicatordata = np.empty((0, input_size+pred_size-1, NUM_OBJS, NUM_OBJS))
    for idx in range(start_idx, end_idx):
        if idx >= video_info.shape[0]:
            break
        vid_idx, img_idx = video_info[idx, 0], video_info[idx, 1]
        video_name, anno_name = video_list[vid_idx], anno_list[vid_idx]

        data = parse_image(video_name, vid_idx, img_idx, seq_size)

        boxes, gt_masks, _, dinfo, sinfo = parse_label(anno_name, vid_idx, img_idx, input_size, pred_size)

        # object shapes
        A = np.where(sinfo)[1] + 1
        object_shapes = np.pad(A, (0, (NUM_OBJS - len(A)) % NUM_OBJS), 'constant')    # 1 for ball, 2 for bar, 3 for jar, 4 for stick, 0 for invalid object

        objectdata = np.append(objectdata, np.expand_dims(object_shapes, axis=0), axis=0)
        # ****************************************************** #

        # image flip augmentation
        if random.random() > 0.5 and split == 'train':
            boxes[..., [0, 2]] = input_width - boxes[..., [2, 0]]
            data = np.ascontiguousarray(data[..., ::-1])
            gt_masks = np.ascontiguousarray(gt_masks[..., ::-1])

        # get information about which objects are static and which are dynamic
        num_objs = boxes.shape[1]
        dynamics = np.zeros(NUM_OBJS, dtype=int)
        dynamics[:num_objs] = dinfo

        valid = np.ones(NUM_OBJS)
        valid[num_objs:] = 0
        boxes = np.concatenate([boxes] + [boxes[:, :1] for _ in range(NUM_OBJS - num_objs)], axis=1)

        # data & rois
        roisv = boxes[:input_size+pred_size-1].copy()
        # rois = boxes[:input_size].copy()
        datav = data[:input_size+pred_size-1].copy()
        # data = data[:input_size].copy()

        # RPCIN NW: object-based 'valid' masks
        shapes = np.zeros((NUM_OBJS, 4))    # 4 object types (ball, bar, jar, stick)
        shapes[:num_objs] = sinfo
        module_valid = np.zeros((input_size+pred_size-1, 6, 6))
        # over timesteps
        for time, (d, b) in enumerate(zip(datav, roisv)):
            data, rois = np.expand_dims(d, axis=0), np.expand_dims(b, axis=0)

            # dual condition
            # for every valid object, check its closeness with every other valid object
            for obj1 in range(num_objs):
                for obj2 in range(num_objs):
                    if obj2 == obj1: continue
                    touch = check_overlap(data[0], boxes[0][obj1], boxes[0][obj2], mask_thresh, 0, obj1, obj2)
                    module_valid[time][obj1][obj2] = touch

            # singular condition
            for obj in range(num_objs):
                module_name = number_to_obj_mapping[object_shapes[obj]]
                check = singular_module(data, boxes, valid, shapes, dynamics, mask_thresh, input_size, module_name)
                module_valid[time][obj][obj] = check[obj]

        gtindicatordata = np.append(gtindicatordata, np.expand_dims(module_valid, axis=0), axis=0)
    np.save(fname_obj, objectdata)
    np.save(fname_gtindicator, gtindicatordata)


if __name__ == "__main__":
    # ********************************************************************************** #
    # ***************************** INPUT=3, OUTPUT=1 ********************************** #
    # total range for TRAIN: 2568138                                                     #
    # do in 10 batches - gap=1000                                                        #
    # (0, 257k), (257k, 514k), (514k, 771k), (771k, 1028k), (1028k, 1285k)               #
    # (1285k, 1542k), (1542k, 1799k), (1799k, 2056k), (2056k, 2313k), (2313k, 2570k)     #
    #                                                                                    #
    # total range for TEST: 642372                                                       #
    # do in 10 batches - gap=1000                                                        #
    # (0, 65k), (65k, 130k), (130k, 195k), (195k, 260k), (260k, 325k)                    #
    # (325k, 390k), (390k, 455k), (455k, 520k), (520k, 585k), (585k, 650k)               #
    # ********************************************************************************** #

    # ********************************************************************************** #
    # ***************************** INPUT=1, OUTPUT=5 /10 ****************************** #
    # total range for TRAIN: 198958                                                      #
    # do in 10 batches - gap=20k                                                         #
    # (0, 20k), (20k, 40k), (40k, 80k), (80k, 100k), (100k, 120k)                        #
    # (120k, 140k), (140k, 160k), (160k, 180k), (180k, 200k)                             #
    #                                                                                    #
    # total range for TEST: 49726                                                        #
    # do in 5 batches - gap=10k                                                          #
    # (0, 10k), (10k, 20k), (20k, 30k), (30k, 40k), (40k, 50k)                           #
    # ********************************************************************************** #

    args = arg_parse()
    split = args.split
    mask_thresh = args.mask_thresh
    input_size, pred_size = args.input_size, args.pred_size

    # video annotations
    env_list = open(DATA + 'within_'+args.split+'_fold_0.txt', 'r').read().split('\n')
    video_list = sum([sorted(glob(f'{DATA}/images/{env.replace(":", "/")}/*.npy')) for env in env_list], [])
    anno_list = [(v[:-4] + '_boxes.hkl').replace('images', 'labels') for v in video_list]
    video_info_name = DATA+'within_'+args.split+'_'+str(input_size)+'_'+str(pred_size)+'_fold_0_info.npy'
    video_info = np.load(video_info_name)

    # create directory
    if not os.path.exists(DATA+'/thresh/'+str(mask_thresh)):
        os.makedirs(DATA+'/thresh/'+str(mask_thresh))

    gap = 1000
    curr_idx_tuples = [(i, i+gap) for i in range(args.start, args.end, gap)]
    Parallel(n_jobs=15)(delayed(saveinfo)(video_info, anno_list, video_list, start_idx, end_idx, input_size, pred_size, mask_thresh) for (start_idx, end_idx) in curr_idx_tuples)

    # # temp purva testing
    # saveinfo(video_info, anno_list, video_list, args.start, args.end, input_size, pred_size, mask_thresh)