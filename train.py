r"""
python train.py \
--cfg configs/phyre/pred/rpcin.yaml \
--gpus 2 \
--conv 1 \
--trans 0 \
--ind 1.0 \
--output conv_attn_valid_6_relresidual_afflayernorm_ind_1.0_bce

TODO: change train code to account for dataloader changes
"""
import os
import torch
import random
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from neuralphys.datasets import *
from neuralphys.utils.config import _C as cfg
from neuralphys.utils.logger import setup_logger, git_diff_config
from neuralphys.models import *
from neuralphys.trainer import Trainer


def arg_parse():
    # only the most general argument is passed here
    # task-specific parameters should be passed by config file
    parser = argparse.ArgumentParser(description='RPIN parameters')
    parser.add_argument('--cfg', required=True, help='path to config file', type=str)
    parser.add_argument('--init', type=str, default='')
    parser.add_argument('--gpus', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--seed', type=int, default=0)
    # transformer parameters
    parser.add_argument('--trans', type=int, default=0)
    parser.add_argument('--conv', type=int, default=0)
    # indicator alignment loss
    parser.add_argument('--ind', type=float, default=0, help='indicator alignment loss weight')
    return parser.parse_args()


def main():
    # the wrapper file contains:
    # 1. setup training environment
    # 2. setup config
    # 3. setup logger
    # 4. setup model
    # 5. setup optimizer
    # 6. setup dataset

    # ---- setup training environment
    args = arg_parse()
    rng_seed = args.seed
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        num_gpus = torch.cuda.device_count()
    else:
        assert NotImplementedError

    # ---- setup config files
    cfg.merge_from_file(args.cfg)
    cfg.SOLVER.BATCH_SIZE *= num_gpus
    cfg.SOLVER.BASE_LR *= num_gpus
    cfg.freeze()
    output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATA_ROOT.split('/')[2], args.output)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(args.cfg, os.path.join(output_dir, 'config.yaml'))
    shutil.copy(os.path.join('neuralphys/models/', cfg.RIN.ARCH + '.py'), os.path.join(output_dir, 'arch.py'))

    # ---- setup logger
    logger = setup_logger('RPIN', output_dir)
    print(git_diff_config(args.cfg))

    model = eval(cfg.RIN.ARCH + '.Net')(trans=args.trans, conv=args.conv)
    model.to(torch.device('cuda'))
    model = torch.nn.DataParallel(
        model, device_ids=list(range(args.gpus.count(',') + 1))
    )

    # ---- setup optimizer, optimizer is not changed
    vae_params = [p for p_name, p in model.named_parameters() if 'vae_lstm' in p_name]
    other_params = [p for p_name, p in model.named_parameters() if 'vae_lstm' not in p_name]
    optim = torch.optim.Adam(
        [{'params': vae_params, 'weight_decay': 0.0}, {'params': other_params}],
        lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    # ---- if resume experiments, use --init ${model_name}
    if args.init:
        logger.info(f'loading pretrained model from {args.init}')
        cp = torch.load(args.init)
        model.load_state_dict(cp['model'], False)

    # ---- setup dataset in the last, and avoid non-deterministic in data shuffling order
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    train_set = eval(f'{cfg.DATASET_ABS}')(data_root=cfg.DATA_ROOT, split='train', image_ext=cfg.RIN.IMAGE_EXT)
    val_set = eval(f'{cfg.DATASET_ABS}')(data_root=cfg.DATA_ROOT, split='test', image_ext=cfg.RIN.IMAGE_EXT)
    kwargs = {'pin_memory': True, 'num_workers': 16}
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1 if cfg.RIN.VAE else cfg.SOLVER.BATCH_SIZE, shuffle=False, **kwargs,
    )
    print(f'size: train {len(train_loader)} / test {len(val_loader)}')

    # ---- setup trainer
    kwargs = {'device': torch.device('cuda'),
              'model': model,
              'optim': optim,
              'ind': args.ind,
              'train_loader': train_loader,
              'val_loader': val_loader,
              'output_dir': output_dir,
              'logger': logger,
              'num_gpus': num_gpus,
              'max_iters': cfg.SOLVER.MAX_ITERS}
    trainer = Trainer(**kwargs)

    try:
        trainer.train()
    except BaseException:
        if len(glob(f"{output_dir}/*.tar")) < 1:
            shutil.rmtree(output_dir)
        raise


if __name__ == '__main__':
    main()
