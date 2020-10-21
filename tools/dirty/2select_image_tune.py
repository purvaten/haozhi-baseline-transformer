import os
import random
import argparse
from glob import glob
from shutil import copyfile


def arg_parse():
    parser = argparse.ArgumentParser(description='randomly select several images')
    parser.add_argument('-i', '--input', required=True, type=str, help='Path to input directory')
    parser.add_argument('-o', '--output', required=True, type=str, help='Path to output directory')
    parser.add_argument('--k', required=True, type=int, help='Number of images to be extracted per folder')
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    root_dir = args.input
    save_dir = args.output
    os.makedirs(save_dir, exist_ok=True)
    file_list = glob(os.path.join(root_dir, '*/'))
    for file_name in file_list:
        image_list = glob(os.path.join(file_name, '*.jpg'))
        # random sample k images from a folder
        image_list = random.choices(image_list, k=args.k)
        for image_name in image_list:
            # -2 is because the last / is the folder identified used in glob
            save_name = file_name.split('/')[-2] + '_' + image_name.split('/')[-1]
            copyfile(image_name, os.path.join(save_dir, save_name))


if __name__ == '__main__':
    main()
