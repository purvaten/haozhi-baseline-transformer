import os
import pdb
import phyre
import torch
import hickle
import numpy as np
from glob import glob

from neuralphys.datasets.phys import Phys
from neuralphys.utils.misc import tprint
from neuralphys.utils.config import _C as C

plot = False  # this is promised to be a temporary flag


class PHYRE(Phys):
    def __init__(self, data_root, split, protocal='within', image_ext='.jpg'):
        super().__init__(data_root, split, image_ext)

        env_list = open(f'{data_root}/{protocal}_{split}_fold_0.txt', 'r').read().split('\n')
        self.video_list = sum([sorted(glob(f'{data_root}/images/{env.replace(":", "/")}/*.npy')) for env in env_list], [])
        self.anno_list = [(v[:-4] + '_boxes.hkl').replace('images', 'labels') for v in self.video_list]

        # just for plot images
        if plot:
            self.video_list = [k for k in self.video_list if int(k.split('/')[-1].split('.')[0]) < 40]
            self.anno_list = [k for k in self.anno_list if int(k.split('/')[-1].split('_')[0]) < 40]
            assert len(self.video_list) == len(self.anno_list)
            self.video_list = self.video_list[::80]
            self.anno_list = self.anno_list[::80]

        # video_info_name = f'for_plot.npy'
        video_info_name = f'{data_root}/{protocal}_{split}_{self.input_size}_{self.pred_size}_fold_0_info.npy'
        if os.path.exists(video_info_name):
            print(f'loading info from: {video_info_name}')
            self.video_info = np.load(video_info_name)
        else:
            self.video_info = np.zeros((0, 2), dtype=np.int32)
            for idx, video_name in enumerate(self.video_list):
                tprint(f'loading progress: {idx}/{len(self.video_list)}')
                num_im = hickle.load(video_name.replace('images', 'labels').replace('.npy', '_boxes.hkl')).shape[0]
                num_sw = num_im - self.seq_size + 1  # number of sliding windows
                if plot:
                    num_sw = 1
                if self.input_size == 1:
                    num_sw = min(1, num_im - self.seq_size + 1)

                if num_sw <= 0:
                    continue
                video_info_t = np.zeros((num_sw, 2), dtype=np.int32)
                video_info_t[:, 0] = idx  # video index
                video_info_t[:, 1] = np.arange(num_sw)  # sliding window index
                self.video_info = np.vstack((self.video_info, video_info_t))
            np.save(video_info_name, self.video_info)

        for module in C.SINGULAR_MODULES:
            self.module_dict[module] = np.load(f'{data_root}/{module}/thresh_{C.MASK_THRESH}/{protocal}_{split}_{self.input_size}_{self.pred_size}_fold_0_info.npy')
        for module in C.DUAL_MODULES:
            self.module_dict[module] = np.load(f'{data_root}/{module}/{C.DUAL_DATATYPE}/thresh_{C.MASK_THRESH}/{protocal}_{split}_{self.input_size}_{self.pred_size}_fold_0_info.npy')

        # load GT indicators and object info tags
        self.objinfo = np.load(f'{data_root}/thresh/{C.MASK_THRESH}/{protocal}_{split}_{self.input_size}_{self.pred_size}_fold_0_objinfo.npy')
        self.gtindicatorinfo = np.load(f'{data_root}/thresh/{C.MASK_THRESH}/{protocal}_{split}_{self.input_size}_{self.pred_size}_fold_0_gtindicatorinfo.npy')


    def _parse_image(self, video_name, vid_idx, img_idx):
        if C.INPUT.PHYRE_USE_EMBEDDING:
            data = np.load(video_name)[::-1]
            data = np.ascontiguousarray(data)
            data = torch.from_numpy(data).long()
            data = self._image_colors_to_onehot(data)
            data = data.numpy()[None]
        else:
            env_name = video_name.split('/')[-3]
            images = hickle.load(video_name.replace('images', 'full').replace('.npy', '_image.hkl'))
            data = np.array([phyre.observations_to_float_rgb(img.astype(int)) for img in images], dtype=np.float).transpose((0, 3, 1, 2))
            data = data[img_idx:img_idx + self.seq_size]
        return data, images[img_idx:img_idx + self.seq_size], env_name

    def _parse_label(self, anno_name, vid_idx, img_idx):
        boxes = hickle.load(anno_name)[img_idx:img_idx + self.seq_size, :, 1:]
        gt_masks = np.zeros((self.pred_size, boxes.shape[1], C.RIN.MASK_SIZE, C.RIN.MASK_SIZE))
        if C.RIN.MASK_LOSS_WEIGHT > 0:
            anno_name = anno_name.replace('boxes.', 'masks.')
            gt_masks = hickle.load(anno_name)
            gt_masks = gt_masks[img_idx:img_idx + self.seq_size].astype(np.float32)
            gt_masks = gt_masks[self.input_size:]

        labels = torch.zeros(1)
        if C.RIN.SEQ_CLS_LOSS_WEIGHT > 0:
            labels[:] = hickle.load(anno_name.replace('_boxes.', '_label.'))

        if plot:
            boxes = np.concatenate([boxes] + [boxes[[-1]] for _ in range(self.seq_size - boxes.shape[0])], axis=0)
            gt_masks = np.concatenate(
                [gt_masks] + [gt_masks[[-1]] for _ in range(self.pred_size - gt_masks.shape[0])], axis=0
            )

        return boxes, gt_masks, labels

    @staticmethod
    def _image_colors_to_onehot(indices):
        onehot = torch.nn.functional.embedding(
            indices, torch.eye(phyre.NUM_COLORS, device=indices.device))
        onehot = onehot.permute(2, 0, 1).contiguous()
        return onehot
