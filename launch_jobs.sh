#!/bin/bash
#SBATCH -e logs/slurm/err/test.err
#SBATCH -o logs/slurm/out/test.out
#SBATCH -p short
#SBATCH --gres gpu:0
#SBATCH -J test
#SBATCH -c 15

# # TRAIN
# srun /srv/share/purva/conda3/envs/rpin/bin/python3.6 tools/gen_module_masks.py \
# --split 'train' \
# --input_size 1 \
# --pred_size 5 \
# --mask_thresh 1.0 \
# --start ${1} \
# --end ${2}

# TEST
srun /srv/share/purva/conda3/envs/rpin/bin/python3.6 tools/gen_module_masks.py \
--split 'test' \
--input_size 1 \
--pred_size 10 \
--mask_thresh 1.0 \
--start ${1} \
--end ${2}
