import os
import pickle
import numpy as np
from matplotlib import pyplot as plt


def enumerate_actions():
    sample_act_theta = np.arange(0, 36) / 36.0 * np.pi * 2
    sample_act_vel = np.array([2.5, 3.75, 5.0, 6.25, 7.5])
    act_mesh = np.meshgrid(sample_act_theta, sample_act_vel)
    actions = np.hstack([np.sin(act_mesh[0].reshape(-1, 1)),
                         np.cos(act_mesh[0].reshape(-1, 1)),
                         act_mesh[1].reshape(-1, 1)])
    return actions


def simulation_rollout(data, v, im_h, im_w):
    def new_speeds(m1, m2, v1, v2):
        new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
        new_v1 = new_v2 + (v2 - v1)
        return new_v1, new_v2

    rollout_length = 200   # self.actor_rollout + 1 + 1
    n = data.shape[0]

    X = np.zeros((rollout_length, n, 2), dtype='float')
    y = np.zeros((rollout_length, n, 2), dtype='float')

    eps = 0.5
    friction = 0.99
    x = data[:, :2].copy()
    v = v.copy()
    r = 1.5 * np.ones((n,))
    size = [im_h, im_w]
    hit_frame = -1
    for t in range(rollout_length):

        for i in range(n):
            X[t, i] = x[i]
            y[t, i] = v[i]

        if len(np.where(np.abs(y).sum((0, 2)) < 1e-5)[0]) == 0:
            hit_frame = t
            break

        for mu in range(int(1 / eps)):
            for i in range(n):
                x[i] += eps * v[i]
            for i in range(n):
                for z in range(2):
                    if x[i][z] - r[i] < 0:
                        v[i][z] = abs(v[i][z])  # want positive
                    if x[i][z] + r[i] > size[z]:
                        v[i][z] = -abs(v[i][z])  # want negative

            for i in range(n):
                for j in range(i):
                    if np.linalg.norm(x[i] - x[j]) < r[i] + r[j]:
                        # the bouncing off part:
                        w = x[i] - x[j]
                        w = w / np.linalg.norm(w)

                        v_i = np.dot(w.transpose(), v[i])
                        v_j = np.dot(w.transpose(), v[j])

                        new_v_i, new_v_j = new_speeds(1, 1, v_i, v_j)

                        v[i] += w * (new_v_i - v_i)
                        v[j] += w * (new_v_j - v_j)
        v *= friction

    return hit_frame, X, y


def ar(x, y, z):
    return z / 2 + np.arange(x, y, z, dtype='float')


def draw_image(X, res, r=None):
    T, n = np.shape(X)[0:2]
    A = np.zeros((X.shape[0], res[0], res[1], 3), dtype='float')

    [X_perm, Y_perm] = np.meshgrid(ar(0, 1, 1. / res[1]) * res[1], ar(0, 1, 1. / res[0]) * res[0])
    X_perm += 0.5
    Y_perm += 0.5
    colors = [[0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]]
    for t in range(X.shape[0]):
        for i in range(n):
            A[t, :, :, 0] += colors[i][0] * np.exp(-(((Y_perm - X[t, i, 0]) ** 2 +
                                    (X_perm - X[t, i, 1]) ** 2) /
                                   (r[i] ** 2)) ** 4)
            A[t, :, :, 1] += colors[i][1] * np.exp(-(((Y_perm - X[t, i, 0]) ** 2 +
                                    (X_perm - X[t, i, 1]) ** 2) /
                                   (r[i] ** 2)) ** 4)
            A[t, :, :, 2] += colors[i][2] * np.exp(-(((Y_perm - X[t, i, 0]) ** 2 +
                                    (X_perm - X[t, i, 1]) ** 2) /
                                   (r[i] ** 2)) ** 4)

        A[t][A[t] > 1] = 1
    return A


def show_sample(V, i):
    logdir = f'./debug/img_{i}'
    os.makedirs(logdir, exist_ok=True)
    T = V.shape[0]
    for t in range(T):
        plt.imshow(V[t])
        # Save it
        fname = logdir + '/' + str(t) + '.png'
        plt.savefig(fname)


def main():
    data_root = 'data'
    dataset_name = 'vinv11'
    src_pkl = f'{data_root}/{dataset_name}/test.pkl'
    with open(src_pkl, 'rb') as f:
        data = pickle.load(f)
    actions = data['a']
    images = data['X']
    data = data['y']
    num_objs = 3
    num_samples, sample_length = data.shape[:2]
    im_h, im_w = images.shape[2], images.shape[3]
    valid_idx = []
    min_hits = []
    for i in range(num_samples):
        min_hit = 100000
        actions = enumerate_actions()
        hits = []
        for action in actions:
            for obj_id in range(num_objs):
                # get initial configurations of the table
                action_array = np.zeros((num_objs, 3), dtype=np.float32)
                action_array[obj_id, :] = action
                v = (action_array[:, :2] * action_array[:, 2:]).copy()
                hit, X, y = simulation_rollout(data[i, 0], v, im_h, im_w)
                # print(hit)
                # if 0 < hit < 10:
                #     size_h = 48
                #     size_w = 96
                #     res = [size_h, size_w]
                #     X = X[:10]
                #     y = y[:10]
                #     V = draw_image(X, res, np.array([2.5] * num_objs)).reshape((X.shape[0], res[0], res[1], 3))
                #     y = np.concatenate((X, y), axis=2)
                #     show_sample(V, len(hits))
                #     hits.append(hit)
                min_hit = min(hit, min_hit) if hit > 0 else min_hit

        print(f'for sample {i}, min hit is: {min_hit}')
        if min_hit < 10000:
            valid_idx.append(i)
            min_hits.append(min_hit)


if __name__ == '__main__':
    main()
