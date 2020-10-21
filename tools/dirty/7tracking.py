import os
import matplotlib.pyplot as plt
import cv2
from glob import glob
import pickle
import argparse
import warnings
import numpy as np


def arg_parse():
    parser = argparse.ArgumentParser(description='get instance id from bounding box overlap')
    parser.add_argument('--src', required=True, type=str, help='Path to input directory')
    parser.add_argument('--num-objects', type=int, default=3, help='number of objects')
    parser.add_argument('--smooth-window', type=int, default=1, help='smooth multiple frame overlap')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args


def get_cropped_feat(boxes, feat):
    rst = []
    for b in boxes:
        b = np.round(b).astype(int)
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        f = feat[y1:y2, x1:x2, :].mean(axis=(0, 1))
        rst.append(f / np.linalg.norm(f))
    return np.array(rst)


def template_similarity(boxes, query_boxes):
    """
    determine overlaps between boxes and query_boxes
    :param boxes: n * 4 bounding boxes
    :param query_boxes: k * 4 bounding boxes
    :return: overlaps: n * k overlaps
    """
    n = boxes.shape[0]
    k = query_boxes.shape[0]
    overlaps = np.zeros((n, k), dtype=np.float32)
    for n_ in range(n):
        for k_ in range(k):
            overlaps[n_, k_] = np.abs(boxes[n_] - query_boxes[k_]).mean()
    return overlaps


def overlap_similarity(boxes, query_boxes):
    """
    determine overlaps between boxes and query_boxes
    :param boxes: n * 4 bounding boxes
    :param query_boxes: k * 4 bounding boxes
    :return: overlaps: n * k overlaps
    """
    n_ = boxes.shape[0]
    k_ = query_boxes.shape[0]
    distances = np.zeros((n_, k_), dtype=np.float)
    for n in range(n_):
        xc1 = (boxes[n, 0] + boxes[n, 2]) / 2
        yc1 = (boxes[n, 1] + boxes[n, 3]) / 2
        for k in range(k_):
            xc2 = (query_boxes[k, 0] + query_boxes[k, 2]) / 2
            yc2 = (query_boxes[k, 1] + query_boxes[k, 3]) / 2
            distances[n, k] = (xc1 - xc2) ** 2 + (yc1 - yc2) ** 2
    return distances


def main():
    args = arg_parse()
    anno_list = sorted(glob(args.src + '*[0-9].pkl'))
    video_list = sorted(glob(os.path.join(args.src, '*/')))
    for k, anno in enumerate(anno_list):
        print(k)
        with open(anno, 'rb') as f:
            boxes = pickle.load(f)
        if not (boxes.sum(axis=2).reshape(-1) > 1e-5).all():
            print(f'find zero box: {k}, {np.where(boxes.sum(axis=2).reshape(-1) < 1e-5)[0]}, {len(boxes) * 3}')
        if (boxes == -1).any():
            print('finding negative box')
        num_frames = boxes.shape[0]
        processed = np.zeros((num_frames, args.num_objects, 5))
        image_list = sorted(glob(os.path.join(video_list[k], '*.jpg')))
        feats = np.zeros((num_frames, args.num_objects, 3))
        # when i is 0, we initialize the bounding boxes
        # warnings.warn('you are clipping images')
        feat = cv2.imread(image_list[0])  # [70:400, 80:800, :]
        feats[0, :, :] = get_cropped_feat(boxes[0, :], feat)
        for i in range(args.num_objects):
            processed[0, i, 1:] = boxes[0, i, :]
            processed[0, i, 0] = i

        for i in range(1, num_frames):
            # warnings.warn('check whether you did the cropping for dist3cu')
            feat = cv2.imread(image_list[i])  # [70:400, 80:800, :]
            cur_feat = get_cropped_feat(boxes[i, :], feat)
            ovs = template_similarity(cur_feat, feats[i - 1, :])
            # previous frame index:
            prev_obj_id = processed[i - 1, :, 0]
            idx = ovs.argmin(axis=1)
            if len(np.unique(idx)) != args.num_objects:
                # here we impose the bounding box overlap rule to see whether it can solve the problem
                distance = overlap_similarity(boxes[i, :], processed[i - 1, :, 1:])
                idx = distance.argmin(axis=1)
                if len(np.unique(idx)) != args.num_objects:
                    # more complex logic:
                    idx = -np.ones((args.num_objects,), dtype=np.int)
                    while not (distance == np.inf).all():
                        # first calculate the global min
                        mask_idx = distance.min(axis=1).argmin()
                        idx[mask_idx] = distance[mask_idx, :].argmin()
                        distance[mask_idx, :] = np.inf
                        distance[:, idx[mask_idx]] = np.inf

            processed[i, :, 0] = prev_obj_id[idx]
            processed[i, :, 1:] = boxes[i]
            feats[i, :] = cur_feat

            if args.debug:
                print('boxes')
                print(boxes[i])
                print('feats')
                print(feats[i])
                print('overlaps')
                print(ovs)
                print(idx)

                plt.close()
                plt.subplot(2, 1, 2)
                plt.imshow(feat[..., ::-1])
                for b, f in zip(boxes[i], feats[i]):
                    rect = plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], color='white', fill=False)
                    plt.gca().add_patch(rect)
                    plt.gca().text(b[0], b[1] - 2, f'{f[0]:.2f} {f[1]:.2f} {f[2]:.2f}', fontsize=6, color='white')
                plt.subplot(2, 1, 1)
                plt.imshow(cv2.imread(image_list[i - 1])[..., ::-1])
                for b, f in zip(processed[i - 1], feats[i - 1]):
                    inst_id = int(b[0])
                    b = b[1:]
                    rect = plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], color='white', fill=False)
                    plt.gca().add_patch(rect)
                    plt.gca().text(b[0] - 10, b[1] - 25, f'{inst_id:d}', fontsize=6, color='white')
                    plt.gca().text(b[0] - 10, b[1] - 3, f'{f[0]:.2f} {f[1]:.2f} {f[2]:.2f}', fontsize=6, color='white')
                plt.show()

        with open(anno.split('.')[0] + '_track.pkl', 'wb') as f:
            pickle.dump(processed, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
