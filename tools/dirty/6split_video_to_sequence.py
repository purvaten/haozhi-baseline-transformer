import os
import cv2
from glob import glob
import shutil
import pickle
import argparse
import numpy as np

list_0 = [
    [305, 545],
    [931, 1239],
    [2713, 3035],
    [3352, 3676],
    [4012, 4286],
    [4596, 4926],
    [5327, 5658],
    [5895, 6254],
    [6621, 6944],
    [7292, 7664],
    [7860, 8246],
    [8491, 8807],
    [8985, 9300],
    [9700, 10110],
    [10355, 10777],
    [10983, 11310],
    [11573, 12000],
    [12531, 12715],
    [13585, 13883],
    [14185, 14508],
    [14746, 15100],
    [16231, 16602],
    [16953, 17320],
    [17992, 18358],
    [18645, 18927],
    [19224, 19635],
    [19863, 20200],
    [20449, 20659],
    [21059, 21397],
    [21807, 22097],  # 29
]

list_1 = [
    [360, 592],  # 30
    [1702, 2033],
    [2734, 3041],
    [5477, 5783],
    [6253, 6568],
    [7332, 7527],
    [10312, 10455],
    [11016, 11296],
    [12373, 12693],
    [14914, 15297],
    [16392, 16728],  # 40
    [19407, 19713],
    [20448, 20722],
    [21621, 21946],
    [23530, 23764],
    [25462, 25750],
    [26970, 27345],
    [30049, 30208],
    [32507, 32703],
    [33325, 33618],
    [34012, 34219],  # 50
    [35926, 36257],
    [38838, 39134],
    [39738, 40038],
    [40649, 40811],
]

list_2 = [
    [730, 927],  # 55
    [5491, 5750],
    [15298, 15640],
    [20734, 20918],
    [26843, 27131],
    [32315, 32671],  # 60
    [36964, 37189],
    [50616, 50897],
    [56347, 56580],
    [62196, 62478],
    [67025, 67246],
    [72299, 72535],
    [85187, 85465],  # 67
]


def arg_parse():
    parser = argparse.ArgumentParser(description='randomly select several images')
    # parser.add_argument('-i', '--input', required=True, type=str, help='Path to input directory')
    # parser.add_argument('-o', '--output', required=True, type=str, help='Path to input directory')
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    im_folders = [f'data/ytb/raw_images/{i}/' for i in range(3)]
    anno_file_list = [f'data/ytb/post_images/{i}_new.pkl' for i in range(3)]

    split_frame_list = [list_0, list_1, list_2]
    video_cnt = 0
    for im_folder, anno_file, split_list in zip(im_folders, anno_file_list, split_frame_list):
        with open(anno_file, 'rb') as f:
            anno_dict = pickle.load(f)
        annos = anno_dict['gt_bboxes']
        for (a, b) in split_list:
            dst_dir = f'data/preytbv1/{video_cnt:03d}'
            os.makedirs(dst_dir, exist_ok=True)
            new_anno = np.zeros((0, 3, 4))
            im_id = 0
            for i in range(a, b + 1):
                src_name = f'{im_folder}/{i:06d}.jpg'
                dst_name = f'{dst_dir}/{im_id:03d}.jpg'
                # maybe do resize here?
                shutil.copyfile(src_name, dst_name)
                # for the annotation, the index is offset by one
                new_anno = np.concatenate([new_anno, annos[[i]]], axis=0)
                im_id += 1
            with open(f'data/preytbv1/{video_cnt:03d}.pkl', 'wb') as f:
                pickle.dump(new_anno, f, protocol=pickle.HIGHEST_PROTOCOL)
            video_cnt += 1


if __name__ == '__main__':
    main()
