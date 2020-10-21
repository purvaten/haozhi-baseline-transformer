import os
import cv2
import pickle
import argparse
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def arg_parse():
    parser = argparse.ArgumentParser(description='get instance id from bounding box overlap')
    args = parser.parse_args()
    return args


def get_cropped_feat(b, feat):
    rst = []
    x1, y1, x2, y2 = np.ceil(b[0]), np.ceil(b[1]), np.floor(b[2]), np.floor(b[3])
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    f = feat[y1:y2, x1:x2, :].max(axis=(0, 1))
    return f


def main():
    args = arg_parse()
    split = 'test'
    vis_dir = 'dynamics/ytbv2_color'
    video_list = sorted(glob(f'data/{vis_dir}/{split}/*/'))
    anno_list = sorted(glob(f'data/{vis_dir}/{split}/*[0-9].pkl'))
    for video_name, anno_name in zip(video_list, anno_list):
        image_list = sorted(glob(video_name + '*.jpg'))
        with open(anno_name, 'rb') as f:
            bboxes = pickle.load(f)
        save_dir = video_name.replace('data/ytbv3', 'debug')
        os.makedirs(save_dir, exist_ok=True)
        batch_stats = np.zeros((3, 3))
        for im_id, image_name in enumerate(image_list):
            im = cv2.imread(image_name)
            plt.imshow(im[..., ::-1])
            bbox = bboxes[im_id]
            colors = ['white', 'red', 'blue']
            # print(im_id)
            for i, b in enumerate(bbox):
                inst_id = int(b[0])
                b = b[1:]
                rect = plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], color=colors[i], fill=False)
                plt.gca().add_patch(rect)
                plt.gca().text(b[0], b[1] - 2, f'{inst_id}', fontsize=6, color=colors[i])
                batch_stats[inst_id] += get_cropped_feat(b, im[..., ::-1])
            # plt.show()
            # print(save_dir)
            # plt.savefig(f'{save_dir}/{im_id}.jpg'), plt.close()
        print(batch_stats[0] / len(image_list))
        print(batch_stats[1] / len(image_list))
        print(batch_stats[2] / len(image_list))
        red_idx = batch_stats.min(axis=1).argmin()
        white_idx = batch_stats.sum(axis=1).argmax()
        assert red_idx != white_idx
        yellow_idx = 3 - red_idx - white_idx
        print(white_idx, red_idx, yellow_idx)
        bboxes = bboxes[:, [white_idx, red_idx, yellow_idx], :]
        with open(anno_name, 'wb') as f:
            pickle.dump(bboxes, f, pickle.HIGHEST_PROTOCOL)
        print(anno_name)
        print('-----------------------------')

    # ax.scatter(x, y, z, c='r', marker='o')
    # plt.show()


if __name__ == '__main__':
    main()
