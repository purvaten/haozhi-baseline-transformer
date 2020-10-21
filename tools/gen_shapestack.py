import os
import cv2
import pickle
import shutil
import numpy as np

from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt


if __name__ == '__main__':
    cache_name = 'data/dynamics/shape-stack-v0/ss4/'
    # cache_name = 'debug'
    r = 35
    os.makedirs(cache_name, exist_ok=True)
    data_path = 'data/shapestacks'
    # src_path = os.path.join(data_path, 'splits', 'env_ccs+blocks-hard+easy-h=3-vcom=1+2+3-vpsf=0')
    src_path = os.path.join(data_path, 'splits', 'env_ccs+blocks-hard+easy-h=4-vcom=1+2+3+4+5+6-vpsf=0')
    vid_dir = os.path.join(data_path, 'frc_35')
    datasets = ['eval']
    for dataset in datasets:
        data_list_name = f'{src_path}/{dataset}.txt'
        with open(data_list_name) as f:
            vid_name_list = [line.split()[0] for line in f]
        for vid_id, vid_name in enumerate(tqdm(vid_name_list)):
            vid_path = os.path.join(vid_dir, vid_name)
            image_list = sorted(glob(os.path.join(vid_path, '*.jpg')))
            bboxes = np.load(os.path.join(vid_path, 'cam_1.npy'))
            num_objs = bboxes.shape[1]
            rst_bboxes = np.zeros((bboxes.shape[0], bboxes.shape[1], 5))
            for t in range(bboxes.shape[0]):
                rst_bboxes[t, :, 0] = np.arange(num_objs)
                rst_bboxes[t, :, 1] = bboxes[t, :, 0] * 224 - r
                rst_bboxes[t, :, 2] = bboxes[t, :, 1] * 224 - r
                rst_bboxes[t, :, 3] = bboxes[t, :, 0] * 224 + r
                rst_bboxes[t, :, 4] = bboxes[t, :, 1] * 224 + r
            with open(f'{cache_name}/{vid_id:04d}_boxes.pkl', 'wb') as f:
                pickle.dump(rst_bboxes, f, pickle.HIGHEST_PROTOCOL)

            rst_masks = np.zeros((bboxes.shape[0], bboxes.shape[1], 28, 28))
            target_vid_path = os.path.join(cache_name, f'{vid_id:04d}')
            os.makedirs(target_vid_path, exist_ok=True)
            for im_id, (image_name, bbox) in enumerate(zip(image_list, bboxes)):
                # deal with image
                shutil.copy(image_name, os.path.join(target_vid_path, f'{im_id:03d}.jpg'))
                im = cv2.imread(image_name)

                # plt.figure(figsize=(12, 12))
                # plt.subplot(2, 4, 1)
                # plt.imshow(im[..., ::-1])

                if im_id == 0:
                    obj_colors = []
                    for obj_id in range(num_objs):
                        xc, yc = bbox[obj_id, 0] * 224, bbox[obj_id, 1] * 224
                        xc = max(min(xc, 224 - 1), 0)
                        yc = max(min(yc, 224 - 1), 0)
                        obj_color = im[int(yc), int(xc), :].astype(np.float32)
                        obj_colors.append(obj_color)

                for obj_id in range(num_objs):
                    xc, yc = bbox[obj_id, 0] * 224, bbox[obj_id, 1] * 224
                    xc = max(min(xc, 224 - 1), 0)
                    yc = max(min(yc, 224 - 1), 0)
                    obj_color = obj_colors[obj_id]
                    obj_mask = ((im.astype(np.float32) - obj_color[None, None]) ** 2).sum(2) >= 8000
                    plt_im = im.copy()
                    plt_im[obj_mask] = 0
                    im_mask = ((im.astype(np.float32) - obj_color[None, None]) ** 2).sum(2) < 8000
                    im_mask = np.pad(im_mask, 2 * r)
                    if xc < -r or yc < -r or xc > 224 + r or yc > 224 + r:
                        obj_mask = np.zeros((28, 28))
                    else:
                        xc += 2 * r
                        yc += 2 * r
                        x1, x2 = xc - r, xc + r
                        y1, y2 = yc - r, yc + r
                        obj_mask = im_mask[int(y1):int(y2), int(x1):int(x2)]
                        obj_mask = cv2.resize(obj_mask.astype(np.float32), (28, 28)) >= 0.5
                    rst_masks[im_id, obj_id] = obj_mask
                    # plt.subplot(2, 4, 2 + obj_id)
                    # plt.imshow(plt_im[..., ::-1])
                    # plt.subplot(2, 4, 6 + obj_id)
                    # plt.imshow(obj_mask)

                # plt.savefig(f'debug/{vid_id}_{im_id}.png', format='png')
                # plt.close()

            with open(f'{cache_name}/{vid_id:04d}_masks.pkl', 'wb') as f:
                pickle.dump(rst_masks, f, pickle.HIGHEST_PROTOCOL)
