import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
plt.rcParams['figure.figsize'] = (48, 16)
# global variables for config
fix_keyframe = False
fix_keyframe_step = 5
# PHYRE setting
# num_objs = 2
# data_dir = 'data/PHYREv1/test/*[0-9].pkl'
# ytb setting
num_objs = 3
data_dir = 'data/dynamics/ytbv2/train/*[0-9].pkl'
eps = 0.5 if 'ytb' in data_dir else 1e-3
init_eps = 1.0 if 'ytb' in data_dir else 0.1
angle_thresh = 0.85 if 'ytb' in data_dir else 0.95


def main():
    pkl_names = sorted(glob(data_dir))
    all_key_frames = []
    all_col_frames = []
    for pkl_name in pkl_names:
        with open(pkl_name, 'rb') as f:
            bboxes = pickle.load(f)
        prev_bboxes = bboxes[:-2]
        cur_bboxes = bboxes[1:-1]
        next_bboxes = bboxes[2:]
        idx = 1
        key_frames = []
        key_frames_traj = [[False for _ in range(num_objs)]]
        col_frames = []
        for prev_bbox, cur_bbox, next_bbox in zip(prev_bboxes, cur_bboxes, next_bboxes):
            prev_bbox = np.vstack([prev_bbox[:, 1] + prev_bbox[:, 3], prev_bbox[:, 2] + prev_bbox[:, 4]]).transpose() / 2
            cur_bbox = np.vstack([cur_bbox[:, 1] + cur_bbox[:, 3], cur_bbox[:, 2] + cur_bbox[:, 4]]).transpose() / 2
            next_bbox = np.vstack([next_bbox[:, 1] + next_bbox[:, 3], next_bbox[:, 2] + next_bbox[:, 4]]).transpose() / 2
            v1 = next_bbox - cur_bbox
            v1_mag = np.linalg.norm(v1, axis=1)
            v1 = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-5)
            v0 = cur_bbox - prev_bbox
            v0_mag = np.linalg.norm(v0, axis=1)
            v0 = v0 / (np.linalg.norm(v0, axis=1, keepdims=True) + 1e-5)

            key_frame = [False for _ in range(num_objs)]
            col_frame = False

            for i in range(num_objs):
                # stable
                if v1_mag[i] < eps:
                    continue

                # case: two ball hitting, one ball start to move
                if v0_mag[i] < eps and v1_mag[i] - v0_mag[i] > init_eps:
                    key_frame[i] = True
                    col_frame = True

                # hitting something else: need to judge
                if v1[i] @ v0[i].T < angle_thresh and not (v0_mag[i] < eps or v1_mag[i] < eps):
                    # case: hitting ball to cause velocity change:
                    for j in range(num_objs):
                        if i == j:
                            continue
                        if np.sqrt(((cur_bbox[i] - cur_bbox[j]) ** 2).sum()) < 5:
                            col_frame = True
                            key_frame[j] = True

                    key_frame[i] = True

            # here we redefine keyframe
            if fix_keyframe:
                key_frame = True if idx % fix_keyframe_step == 0 else False

            key_frames_traj.append(key_frame)

            if np.sum(key_frame) > 0:
                # a = cv2.imread(pkl_name.replace('.pkl', '/') + f'{idx - 1:03d}.jpg')
                # b = cv2.imread(pkl_name.replace('.pkl', '/') + f'{idx:03d}.jpg')
                # c = cv2.imread(pkl_name.replace('.pkl', '/') + f'{idx + 1:03d}.jpg')
                # plt.clf()
                # plt.subplot(1, 3, 1)
                # plt.imshow(a[..., ::-1])
                # plt.subplot(1, 3, 2)
                # plt.imshow(b[..., ::-1])
                # for i in range(num_objs):
                #     if key_frame[i]:
                #         rect = plt.Rectangle((cur_bbox[i, 0] - 1.5, cur_bbox[i, 1] - 1.5),
                #                              3, 3, fill=True, color=(1.0, 1.0, 1.0))
                #         plt.gca().add_patch(rect)
                # plt.subplot(1, 3, 3)
                # plt.imshow(c[..., ::-1])
                # plt.show()
                if len(key_frames) > 0 and key_frames[-1] == idx - 1:
                    key_frames[-1] = idx
                    key_frames_traj[-2] = [False for _ in range(num_objs)]
                else:
                    key_frames.append(idx)

            if col_frame:
                if len(col_frames) > 0 and col_frames[-1] == idx - 1:
                    col_frames[-1] = idx
                else:
                    col_frames.append(idx)

            idx += 1

        # the last frame is not keyframe
        key_frames_traj.append([False for _ in range(num_objs)])
        all_key_frames.append(len(key_frames))
        assert len(key_frames_traj) == len(bboxes)
        with open(pkl_name.replace('.pkl', '_key.pkl'), 'wb') as f:
            pickle.dump(np.array(key_frames_traj), f, pickle.HIGHEST_PROTOCOL)
        all_col_frames.append(len(col_frames))
        with open(pkl_name.replace('.pkl', '_col.pkl'), 'wb') as f:
            pickle.dump(col_frames, f, pickle.HIGHEST_PROTOCOL)

    print(np.mean(all_key_frames))
    print(np.mean(all_col_frames))


if __name__ == '__main__':
    main()
