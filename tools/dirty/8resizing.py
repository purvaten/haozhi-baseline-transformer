import os
import matplotlib.pyplot as plt
import cv2
import pickle
import argparse
import warnings
import numpy as np
from glob import glob


def arg_parse():
    parser = argparse.ArgumentParser(description='get instance id from bounding box overlap')
    args = parser.parse_args()
    return args


def main():
    input_h = 96
    input_w = 192
    tar_name = 'ytbv3/train'
    args = arg_parse()
    video_list = sorted(glob('data/preytbv1/*/'))
    anno_list = sorted(glob('data/preytbv1/*_track.pkl'))
    for video_name, anno_name in zip(video_list, anno_list):
        os.makedirs(video_name.replace('preytbv1', tar_name), exist_ok=True)
        image_list = sorted(glob(video_name + '*.jpg'))

        # resize boxes
        with open(anno_name, 'rb') as f:
            bboxes = pickle.load(f)

        num_boxes = bboxes.shape[0]
        num_obj = bboxes.shape[1]
        idx = bboxes[:, :, 0].argsort()
        idx = idx + np.arange(num_boxes)[:, None] * num_obj
        bboxes = bboxes.reshape(-1, 5)[idx.flatten(), ...].reshape(num_boxes, num_obj, 5)

        for im_id, image_name in enumerate(image_list):
            im = cv2.imread(image_name)
            im_height, im_width = im.shape[:2]
            resize_h, resize_w = input_h / im_height, input_w / im_width
            im = cv2.resize(im, None, fx=resize_w, fy=resize_h, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(image_name.replace('preytbv1', tar_name), im)

            bboxes[im_id, :, [1, 3]] = bboxes[im_id, :, [1, 3]] * (input_w - 1) / (im_width - 1)
            bboxes[im_id, :, [2, 4]] = bboxes[im_id, :, [2, 4]] * (input_h - 1) / (im_height - 1)

        with open(anno_name.replace('preytbv1', tar_name).replace('_track', ''), 'wb') as f:
            pickle.dump(bboxes, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
