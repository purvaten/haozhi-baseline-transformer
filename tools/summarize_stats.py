import os
from glob import glob
import pickle
import numpy as np


def main():
    real_data()
    sim_data()


def real_data():
    src_files = ['data/ytbv1/train/',
                 'data/ytbv1/test/']
    for src_file in src_files:
        pickle_files = sorted(glob(os.path.join(src_file, '*.pkl')))
        init_v_mag_list = []
        mean_v_mag_list = []
        mean_non_zero_v_mag_list = []
        mean_pos_list = []
        for pickle_file in pickle_files:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
            # remove batch size
            data = data[..., 1:]

            pos_data = data.mean()
            mean_pos_list.append(pos_data)

            data = data[1:] - data[:-1]
            data[..., 0] += data[..., 2]
            data[..., 1] += data[..., 3]
            label_data = data[..., :2] / 2

            init_v = label_data[0]
            init_v_mag = np.linalg.norm(init_v, axis=1)
            init_v_mag = init_v_mag[init_v_mag > 0.1]
            if len(init_v_mag) > 0:
                init_v_mag_list.append(init_v_mag.mean())

            mean_v_mag = np.linalg.norm(label_data, axis=2)
            mean_v_mag = mean_v_mag.mean()
            mean_v_mag_list.append(mean_v_mag)

            mean_v_mag = np.linalg.norm(label_data, axis=2)
            mean_v_mag = mean_v_mag[mean_v_mag > 0.1].mean()
            mean_non_zero_v_mag_list.append(mean_v_mag)

        print(src_file + ' :')
        print('mean position: ', np.mean(mean_pos_list))
        print('initial velocity: ', np.mean(init_v_mag_list), np.std(init_v_mag_list))
        print('mean velocity: ', np.mean(mean_v_mag_list), np.std(mean_v_mag_list))
        print('mean non-zero velocity: ', np.mean(mean_non_zero_v_mag_list), np.std(mean_non_zero_v_mag_list))


def sim_data():
    src_files = ['data/vinv8/train.pkl',
                 'data/vinv8/test.pkl']
    for src_file in src_files:
        with open(src_file, 'rb') as f:
            data = pickle.load(f)
        # shape (N x T x 3 x 4), 4 is (pos_y, pos_x, vel_y, vel_x)
        label_data = data['y']

        init_v_mag_list = []
        mean_v_mag_list = []
        mean_non_zero_v_mag_list = []
        mean_pos_list = []
        for label in label_data:

            pos_data = label[..., :2].mean()
            mean_pos_list.append(pos_data)
            # calculate initial velocity
            init_v = label[0][:, 2:]
            init_v_mag = np.linalg.norm(init_v, axis=1)
            init_v_mag = init_v_mag[init_v_mag != 0]
            init_v_mag_list.append(init_v_mag)
            # calculate mean velocity
            mean_v = label[:, :, 2:]
            mean_v_mag = np.linalg.norm(mean_v, axis=2)
            mean_v_mag = mean_v_mag.mean()
            mean_v_mag_list.append(mean_v_mag)
            # calculate mean non zero velocity
            mean_v = label[:, :, 2:]
            mean_v_mag = np.linalg.norm(mean_v, axis=2)
            mean_v_mag = mean_v_mag[mean_v_mag > 1e-3].mean()
            mean_non_zero_v_mag_list.append(mean_v_mag)
        print(src_file + ' :')
        print('mean position: ', np.mean(mean_pos_list))
        print('initial velocity: ', np.mean(init_v_mag_list), np.std(init_v_mag_list))
        print('mean velocity: ', np.mean(mean_v_mag_list), np.std(mean_v_mag_list))
        print('mean non-zero velocity: ', np.mean(mean_non_zero_v_mag_list), np.std(mean_non_zero_v_mag_list))


if __name__ == '__main__':
    main()
