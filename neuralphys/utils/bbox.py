import torch
import numpy as np
import pdb
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def xyxy_to_rois(boxes, batch, time_step, num_devices):
    # convert input bounding box of format (x1, y1, x2, y2) to network input rois
    # two necessary steps are performed:
    # 1. create batch indexes for roi pooling
    # 2. offset the batch_rois for multi-gpu usage
    if boxes.shape[0] != batch:
        assert boxes.shape[0] == (batch // boxes.shape[2])
    batch, num_objs = boxes.shape[0], boxes.shape[2]
    num_im = batch * time_step
    rois = boxes[:, :time_step]
    batch_rois = np.zeros((num_im, num_objs))
    batch_rois[np.arange(num_im), :] = np.arange(num_im).reshape(num_im, 1)
    batch_rois = torch.tensor(batch_rois.reshape((batch, time_step, -1, 1)), dtype=torch.float32)
    load_list = [batch // num_devices for _ in range(num_devices)]
    extra_loaded_gpus = batch - sum(load_list)
    for i in range(extra_loaded_gpus):
        load_list[i] += 1
    load_list = np.cumsum(load_list)
    for i in range(1, num_devices):
        batch_rois[load_list[i - 1]:load_list[i]] -= (load_list[i - 1] * time_step)
    rois = torch.cat([batch_rois, rois], dim=-1)
    return rois


def xyxy_to_posf(boxes, shape):
    # convert input bounding box of format (x1, y1, x2, y2) to position feature input
    height, width = shape[-2:]
    co_f = np.zeros(boxes.shape[:-1] + (4,))
    co_f[..., [0, 2]] = boxes[..., [0, 2]].numpy() / width
    co_f[..., [1, 3]] = boxes[..., [1, 3]].numpy() / height
    coor_features = torch.from_numpy(co_f.astype(np.float32))
    return coor_features


def xcyc_to_xyxy(boxes, scale_h, scale_w, radius):
    # convert input xc yc (normally output by network)
    # to x1, y1, x2, y2 format, with scale and radius offset
    rois = np.zeros(boxes.shape[:3] + (4,))
    rois[..., 0] = boxes[..., 0] * scale_w - radius
    rois[..., 1] = boxes[..., 1] * scale_h - radius
    rois[..., 2] = boxes[..., 0] * scale_w + radius
    rois[..., 3] = boxes[..., 1] * scale_h + radius
    return rois


def xyxy2xywh(boxes):
    assert boxes.ndim == 2
    w = boxes[:, 2] - boxes[:, 0] + 1.0
    h = boxes[:, 3] - boxes[:, 1] + 1.0
    xc = boxes[:, 0] + 0.5 * (w - 1.0)
    yc = boxes[:, 1] + 0.5 * (h - 1.0)
    return np.vstack([xc, yc, w, h]).transpose()


def xywh2xyxy(boxes):
    assert boxes.ndim == 2
    xc, yc = boxes[:, 0], boxes[:, 1]
    w, h = boxes[:, 2], boxes[:, 3]
    x1, x2 = xc - 0.5 * (w - 1.0), xc + 0.5 * (w - 1.0)
    y1, y2 = yc - 0.5 * (h - 1.0), yc + 0.5 * (h - 1.0)
    return np.vstack([x1, y1, x2, y2]).transpose()

def boxes_overlap(imgs, valid, boxes, thresh):
    """Return boolean list of size boxes.shape[0]
        indicating whether to invoke interaction module or not.
       If any 2 bounding boxes are close within threshold, set True.
    """
    # get IOU of all pairs of objects
    times = boxes.shape[0]
    overlap = np.zeros((times), dtype=bool)
    combinations = list(itertools.combinations(np.where(valid == 1)[0], 2))

    for t in range(times):
        for (i, j) in combinations:
            boxA = boxes[t][i].copy()
            boxB = boxes[t][j].copy()

            # draw a border of (thresh/2) for both boxes
            bord = np.array([-1, -1, 1, 1]) * thresh/2
            boxA += bord
            boxB += bord
            boxA[boxA<0], boxA[boxA>128] = 0, 128
            boxB[boxB<0], boxB[boxB>128] = 0, 128

            # (debugging) display the boxes on the image to be sure?
            # display_img_with_bb(np.transpose(imgs[t], (1, 2, 0)), boxA, boxB, '%d_%d_%d.png' % (t,i,j))

            # get iou
            iou = bb_intersection_over_union(boxA, boxB)
            # if iou is exactly 1.0 it is the same object
            # if iou is 0, it means the objects are not touching or 1 or both are invalid
            if iou != 1.0 and iou != 0:
                overlap[t] = 1
                break

    return torch.from_numpy(overlap.astype(np.int))


def bb_intersection_over_union(boxA, boxB):
    # boxes given as xyxy
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0:
        iou = 0
    else:
        iou = interArea / unionArea
    # return the intersection over union value
    return iou


def display_img_with_bb(im, boxA, boxB, name):
    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    a1, a2, a3, a4 = boxA
    b1, b2, b3, b4 = boxB
    recta = patches.Rectangle((a1,a2),a3-a1,a4-a2,linewidth=1,edgecolor='r',facecolor='none')
    rectb = patches.Rectangle((b1,b2),b3-b1,b4-b2,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(recta)
    ax.add_patch(rectb)
    # plt.savefig('images/' + name)
    plt.savefig(name)
    plt.close()
