import os
import cv2
import pickle
import numpy as np


def main():
    r = 2.0
    num_objs = 3
    name = 'simb'
    name_list = ['train', 'test']
    # for i in range(4):
    #     for j in range(2, 4):
    #         name_list.append(f'{j}ball_60_{i}')
    for split in name_list:
        print(split)
        tar_dir = f'data/dynamics/{name}/{split}/'
        os.makedirs(tar_dir, exist_ok=True)
        vin_data_file = f'data/dynamics/{name}/{split}.pkl'
        with open(vin_data_file, 'rb') as f:
            vin_data = pickle.load(f)
        image_data = vin_data['X']
        label_data = vin_data['y']
        key_frame_data = vin_data['keyf']
        col_frame_data = vin_data['colf']
        action_data = vin_data['a']
        num_seq, im_seq_len, im_h, im_w, num_channel = image_data.shape
        _, label_seq_len, _, _ = label_data.shape
        r = vin_data['r'] if 'r' in vin_data else r * np.ones((num_seq, num_objs))
        for i in range(num_seq):
            print(f'{i}/{num_seq}') if i % 100 == 0 else print('', end='')
            # 1. write out images
            cur_image_data = image_data[i]
            cur_label_data = label_data[i]
            cur_target_dir = os.path.join(tar_dir, f'{i:05d}')
            os.makedirs(cur_target_dir, exist_ok=True)
            cur_target_roi_name = os.path.join(tar_dir, f'{i:05d}.pkl')
            # 2. convert position to bounding box locations
            all_rois = np.zeros((label_seq_len, cur_label_data.shape[1], 5))
            for t in range(im_seq_len):
                cur_image = cur_image_data[t]
                with open(os.path.join(cur_target_dir, f'{t:05d}.pkl'), 'wb') as f:
                    pickle.dump(cur_image * 255, f, pickle.HIGHEST_PROTOCOL)
                cv2.imwrite(os.path.join(cur_target_dir, f'{t:05d}_debug.png'), cur_image * 255)
            for t in range(label_seq_len):
                cur_rois = cur_label_data[t]
                all_rois[t, :, 0] = np.arange(num_objs)
                for obj_id in range(num_objs):
                    all_rois[t, obj_id, 1] = cur_rois[obj_id, 1] - r[i, obj_id]
                    all_rois[t, obj_id, 2] = cur_rois[obj_id, 0] - r[i, obj_id]
                    all_rois[t, obj_id, 3] = cur_rois[obj_id, 1] + r[i, obj_id]
                    all_rois[t, obj_id, 4] = cur_rois[obj_id, 0] + r[i, obj_id]
            with open(cur_target_roi_name, 'wb') as f:
                pickle.dump(all_rois, f, pickle.HIGHEST_PROTOCOL)
            with open(cur_target_roi_name.replace('.pkl', '_key.pkl'), 'wb') as f:
                pickle.dump(key_frame_data[i], f, pickle.HIGHEST_PROTOCOL)
            with open(cur_target_roi_name.replace('.pkl', '_col.pkl'), 'wb') as f:
                pickle.dump(col_frame_data[i], f, pickle.HIGHEST_PROTOCOL)
            with open(cur_target_roi_name.replace('.pkl', '_act.pkl'), 'wb') as f:
                pickle.dump(action_data[i], f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
