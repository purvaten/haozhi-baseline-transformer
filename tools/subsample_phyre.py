import os
import shutil
import pickle
from glob import glob
src_folder = 'data/PHYREv1_backup/test/'
tar_folder = 'data/PHYREv1/test/'
folders = sorted(glob(src_folder + '*/'))
idx = 0
for f in folders:
    print(idx, len(folders))
    debug_list = sorted(glob(f + '*_debug.jpg'))[::2]
    raw_list = sorted(glob(f + '*_raw.jpg'))[::2]
    rgb_list = sorted(glob(f + '*_rgb.jpg'))[::2]
    tar_dir = f.replace('_backup', '')
    os.makedirs(tar_dir, exist_ok=True)
    for a, b, c in zip(debug_list, raw_list, rgb_list):
        shutil.copy(a, a.replace('_backup', ''))
        shutil.copy(b, b.replace('_backup', ''))
        shutil.copy(c, c.replace('_backup', ''))
    pkl_name = f[:-1] + '.pkl'
    with open(pkl_name, 'rb') as f:
        A = pickle.load(f)
    A = A[::2]
    with open(pkl_name.replace('_backup', ''), 'wb') as t:
        pickle.dump(A, t, pickle.HIGHEST_PROTOCOL)
    idx += 1