import os
import cv2
import pickle
import argparse
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


def arg_parse():
    parser = argparse.ArgumentParser(description='get instance id from bounding box overlap')
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    split = 'train'
    vis_dir = 'ytbv3'
    video_list = sorted(glob(f'data/{vis_dir}/{split}/*/'))
    anno_list = sorted(glob(f'data/{vis_dir}/{split}/*.pkl'))
    for video_name, anno_name in zip(video_list, anno_list):
        image_list = sorted(glob(video_name + '*.jpg'))
        with open(anno_name, 'rb') as f:
            bboxes = pickle.load(f)
        save_dir = video_name.replace('data/ytbv3', 'debug')
        os.makedirs(save_dir, exist_ok=True)
        for im_id, image_name in enumerate(image_list):
            im = cv2.imread(image_name)
            plt.imshow(im[..., ::-1])
            bbox = bboxes[im_id]
            colors = ['white', 'red', 'blue']
            for i, b in enumerate(bbox):
                inst_id = int(b[0])
                b = b[1:]
                rect = plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], color=colors[i], fill=False)
                plt.gca().add_patch(rect)
                plt.gca().text(b[0], b[1] - 2, f'{inst_id}', fontsize=6, color=colors[i])
            # plt.show()
            plt.savefig(f'{save_dir}/{im_id}.jpg'), plt.close()


if __name__ == '__main__':
    main()
