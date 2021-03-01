import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset

from neuralphys.utils.config import _C as C
from neuralphys.utils.bbox import xyxy2xywh

plot = False  # this is promised to be a temporary flag
debug = False

import pdb


class Phys(Dataset):
    def __init__(self, data_root, split, image_ext='.jpg'):
        self.data_root = data_root
        self.split = split
        self.image_ext = image_ext
        # 1. define property of input and rollout parameters
        self.input_size = C.RIN.INPUT_SIZE  # number of input images
        self.pred_size = eval(f'C.RIN.PRED_SIZE_{split.upper()}')
        self.seq_size = self.input_size + self.pred_size
        # 2. define model configs
        self.input_height, self.input_width = C.RIN.INPUT_HEIGHT, C.RIN.INPUT_WIDTH
        self.video_list, self.anno_list = None, None
        self.video_info = None
        # 3. module specific
        self.module_dict = {}

    def __len__(self):
        return self.video_info.shape[0]

    def __getitem__(self, idx):
        vid_idx, img_idx = self.video_info[idx, 0], self.video_info[idx, 1]
        video_name, anno_name = self.video_list[vid_idx], self.anno_list[vid_idx]
        if C.RIN.VAE:
            data, data_t = self._parse_image(video_name, vid_idx, img_idx)
        else:
            data, data_t, env_name = self._parse_image(video_name, vid_idx, img_idx)

        boxes, gt_masks, labels = self._parse_label(anno_name, vid_idx, img_idx)

        if debug:
            # debug box output
            for t in range(self.seq_size - 1):
                import matplotlib.pyplot as plt
                plt.imshow(data[0].transpose((1, 2, 0)))
                for o_id in range(boxes.shape[1]):
                    x1, y1, x2, y2 = boxes[t + 1, o_id]
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2, color='r')
                    plt.gca().add_patch(rect)
                plt.savefig(f'debug/{idx:03d}_debug.jpg'), plt.close()
                # debug mask output
                for o_id in range(boxes.shape[1]):
                    mask_im = np.zeros((128, 128))
                    x1, y1, x2, y2 = boxes[t + 1, o_id].astype(np.int)
                    mask = cv2.resize(gt_masks[t, o_id].astype(np.float32), (x2 - x1 + 1, y2 - y1 + 1))
                    mask_im[y1:y2 + 1, x1:x2 + 1] = mask
                    plt.imshow(mask_im)
                    plt.savefig(f'debug/{idx:03d}_{o_id}_debug.jpg'), plt.close()

        # image flip augmentation
        if random.random() > 0.5 and self.split == 'train' and C.RIN.HORIZONTAL_FLIP:
            boxes[..., [0, 2]] = self.input_width - boxes[..., [2, 0]]
            data = np.ascontiguousarray(data[..., ::-1])
            gt_masks = np.ascontiguousarray(gt_masks[..., ::-1])

        if random.random() > 0.5 and self.split == 'train' and C.RIN.VERTICAL_FLIP:
            boxes[..., [1, 3]] = self.input_height - boxes[..., [3, 1]]
            data = np.ascontiguousarray(data[..., ::-1, :])
            gt_masks = np.ascontiguousarray(gt_masks[..., ::-1])

        # when the number of objects is fewer than the max number of objects
        num_objs = boxes.shape[1]
        g_idx = []
        for i in range(C.RIN.NUM_OBJS):
            for j in range(C.RIN.NUM_OBJS):
                if j == i:
                    continue
                g_idx.append([i, j, (i < num_objs) * (j < num_objs)])
        g_idx = np.array(g_idx)

        valid = np.ones(C.RIN.NUM_OBJS)
        valid[num_objs:] = 0
        boxes = np.concatenate([boxes] + [boxes[:, :1] for _ in range(C.RIN.NUM_OBJS - num_objs)], axis=1)
        gt_masks = np.concatenate([gt_masks] + [gt_masks[:, :1] for _ in range(C.RIN.NUM_OBJS - num_objs)], axis=1)

        # rois
        rois = boxes[:self.input_size].copy()
        # data
        data_pred = data[self.input_size:].copy()    # GT future frames
        data = data[:self.input_size].copy()
        data_t = data_t[:self.input_size].copy()
        # gt boxes
        gt_boxes = boxes[self.input_size:].copy()
        gt_boxes = xyxy2xywh(gt_boxes.reshape(-1, 4)).reshape((-1, C.RIN.NUM_OBJS, 4))
        gt_boxes[..., 0::2] /= self.input_width
        gt_boxes[..., 1::2] /= self.input_height
        gt_boxes = gt_boxes.reshape(self.pred_size, -1, 4)

        if C.RIN.ROI_MASKING:
            data = torch.from_numpy(data.astype(np.float32))
            data = data[:, None].repeat(1, C.RIN.NUM_OBJS, 1, 1, 1)
            for i in range(data.shape[0]):  # timestep
                for j in range(data.shape[1]):  # num_objs
                    roi = rois[i, j]
                    x1, y1 = int(np.floor(roi[0])), int(np.floor(roi[1]))
                    x2, y2 = int(np.ceil(roi[2])), int(np.ceil(roi[3]))
                    data[i, j, :, :, :x1] = 0
                    data[i, j, :, :y1, :] = 0
                    data[i, j, :, :, x2:] = 0
                    data[i, j, :, y2:, :] = 0
            data = data.numpy()

        if C.RIN.ROI_CROPPING:
            data_crop = np.zeros((data.shape[0], C.RIN.NUM_OBJS,) + data.shape[1:])
            for i in range(data.shape[0]):
                for j in range(num_objs):
                    roi = rois[i, j]
                    x_c = 0.5 * (roi[0] + roi[2])
                    y_c = 0.5 * (roi[1] + roi[3])
                    image = data[i].transpose((1, 2, 0))
                    if 'PHYRE' in self.data_root:
                        x1, y1 = int(np.floor(roi[0])), int(np.floor(roi[1]))
                        x2, y2 = int(np.ceil(roi[2])), int(np.ceil(roi[3]))
                        data_crop_ = image[y1:y2+1, x1:x2+1, :]
                        data_crop_ = cv2.resize(data_crop_, (self.input_width, self.input_height))
                        data_crop[i, j] = data_crop_.transpose((2, 0, 1))
                    else:
                        r = C.RIN.ROI_CROPPING_R
                        d = 2 * r
                        data_crop_ = np.zeros((d, d))
                        image_pad = np.pad(image, ((d, d), (d, d), (0, 0)))
                        if x_c > -r or y_c > -r or x_c < self.input_width + r or y_c < self.input_height + r:
                            x_c += d
                            y_c += d
                            data_crop_ = image_pad[int(y_c - r):int(y_c + r), int(x_c - r):int(x_c + r), :]
                            assert data_crop_.shape[0] == d
                            assert data_crop_.shape[1] == d
                        data_crop_ = cv2.resize(data_crop_, (self.input_width, self.input_height))
                        data_crop[i, j] = data_crop_.transpose((2, 0, 1))
            data = data_crop.copy()

        # module valid stuff
        ALL_MODULES = C.SINGULAR_MODULES + C.DUAL_MODULES
        module_valid = np.empty((len(ALL_MODULES), self.input_size+self.pred_size-1, C.RIN.NUM_OBJS))
        for mid, module in enumerate(ALL_MODULES):
            module_valid[mid] = np.expand_dims(self.module_dict[module][idx].astype('int'), axis=0)

        data = torch.from_numpy(data.astype(np.float32))
        data_pred = torch.from_numpy(data_pred.astype(np.float32))
        data_t = torch.from_numpy(data_t.astype(np.float32))
        rois = torch.from_numpy(rois.astype(np.float32))
        gt_boxes = torch.from_numpy(gt_boxes.astype(np.float32))
        gt_masks = torch.from_numpy(gt_masks.astype(np.float32))
        valid = torch.from_numpy(valid.astype(np.float32))

        return data, data_pred, data_t, env_name, rois, gt_boxes, gt_masks, valid, module_valid, g_idx, labels

    def _parse_image(self, video_name, vid_idx, img_idx):
        raise NotImplementedError

    def _parse_label(self, anno_name, vid_idx, img_idx):
        raise NotImplementedError
