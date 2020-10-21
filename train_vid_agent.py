import os
import time
import torch
import shutil
import random
import argparse
import numpy as np

from torch.utils.data import DataLoader

from neuralphys.models import dqn as nets
from neuralphys.utils.config import _C as C
from neuralphys.utils.misc import tprint, pprint
from neuralphys.datasets.vid_cls import VidPHYRECls
from neuralphys.utils.logger import setup_logger, git_diff_config


take_idx = [0, 3, 6]
is_gt = [0, 0, 0]
rs = False
rss = 0
rse = 5


def test(data_loader, model):
    p_correct, n_correct, p_total, n_total = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for batch_idx, data_tuple in enumerate(data_loader):
            gt, pred, labels = data_tuple
            labels = labels.squeeze(1).to('cuda')
            data = []
            for i, idx in enumerate(take_idx):
                data.append(gt[:, [idx]] if is_gt[i] else pred[:, [idx]])
            data = torch.cat(data, dim=1)
            data = data.long().to('cuda')
            pred = model(data)
            pred = pred.sigmoid() >= 0.5
            p_correct += ((pred == labels)[labels == 1]).sum().item()
            n_correct += ((pred == labels)[labels == 0]).sum().item()
            p_total += (labels == 1).sum().item()
            n_total += (labels == 0).sum().item()
    return p_correct / p_total, n_correct / n_total


def train(train_loader, test_loader, model, optim, scheduler, logger, output_dir):
    max_iters = C.SOLVER.MAX_ITERS
    model.train()

    losses = []
    acc = [0, 0, 0, 0]
    test_accs = []
    last_time = time.time()
    cur_update = 0
    while True:
        for batch_idx, data_tuple in enumerate(train_loader):
            if cur_update >= max_iters:
                break
            model.train()

            p_gt, p_pred, n_gt, n_pred, labels = data_tuple
            # labels = torch.cat([torch.ones(p_gt.shape[0]), torch.zeros(n_gt.shape[0])]).to('cuda')
            labels = torch.cat([labels[:, 0, 0], labels[:, 0, 1]]).to('cuda')
            p_data = []
            n_data = []
            for i, idx in enumerate(take_idx):
                p_data.append(p_gt[:, [idx]] if is_gt[i] else p_pred[:, [idx]])
                n_data.append(n_gt[:, [idx]] if is_gt[i] else n_pred[:, [idx]])
            p_data = torch.cat(p_data, dim=1)
            n_data = torch.cat(n_data, dim=1)
            data = torch.cat([p_data, n_data])

            data = data.long().to('cuda')
            optim.zero_grad()

            # pred = model(data, acts)
            pred = model(data)
            loss = model.ce_loss(pred, labels)

            pred = pred.sigmoid() >= 0.5
            acc[0] += ((pred == labels)[labels == 1]).sum().item()
            acc[1] += ((pred == labels)[labels == 0]).sum().item()
            acc[2] += (labels == 1).sum().item()
            acc[3] += (labels == 0).sum().item()

            loss.backward()
            optim.step()
            scheduler.step()
            losses.append(loss.mean().item())

            cur_update += 1
            speed = (time.time() - last_time) / cur_update
            eta = (max_iters - cur_update) * speed / 3600
            info = f'Iter: {cur_update} / {max_iters}, eta: {eta:.2f}h ' \
                   f'p acc: {acc[0] / acc[2]:.4f} n acc: {acc[1] / acc[3]:.4f}'
            tprint(info)

            if (cur_update + 1) % C.SOLVER.VAL_INTERVAL == 0:
                pprint(info)
                fpath = os.path.join(output_dir, 'last.ckpt')
                torch.save(
                    dict(
                        model=model.state_dict(), optim=optim.state_dict(), done_batches=cur_update + 1,
                        scheduler=scheduler and scheduler.state_dict(),
                    ), fpath
                )

                p_acc, n_acc = test(test_loader, model)
                test_accs.append([p_acc, n_acc])
                model.train()
                acc = [0, 0, 0, 0]
                for k in range(2):
                    info = ''
                    for test_acc in test_accs:
                        info += f'{test_acc[k] * 100:.1f} / '
                    logger.info(info)


def arg_parse():
    # only the most general argument is passed here
    # task-specific parameters should be passed by config file
    parser = argparse.ArgumentParser(description='PCLS parameters')
    parser.add_argument('--cfg', required=True, type=str)
    parser.add_argument('--gpus', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fuse', type=str, default='gg')
    return parser.parse_args()


def main():
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
    C.merge_from_file(args.cfg)
    C.SOLVER.BATCH_SIZE *= num_gpus
    C.SOLVER.BASE_LR *= num_gpus
    C.freeze()
    data_root = C.DATA_ROOT
    output_dir = os.path.join(C.OUTPUT_DIR, 'cls', C.DATA_ROOT.split('/')[2], args.output)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(args.cfg, os.path.join(output_dir, 'config.yaml'))
    # shutil.copy(os.path.join('neuralphys/models/', C.PCLS.ARCH + '.py'), os.path.join(output_dir, 'arch.py'))

    # ---- setup logger
    logger = setup_logger('PCLS', output_dir)
    print(git_diff_config(args.cfg))

    device = 'cuda'
    if C.PCLS.ARCH == 'resnet18':
        model = nets.ResNet18FilmAction(3, fusion_place='last', action_hidden_size=256, action_layers=1)
    elif C.PCLS.ARCH == 'resnet18film':
        model = nets.ResNet18Film(len(take_idx), args.fuse)
    else:
        raise NotImplementedError
    model.to(device)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=C.SOLVER.BASE_LR,
        weight_decay=C.SOLVER.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=C.SOLVER.MAX_ITERS)

    # ---- setup dataset in the last, and avoid non-deterministic in data shuffling order
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    kwargs = {'pin_memory': True, 'num_workers': 16}
    train_set = VidPHYRECls(data_root=data_root, split='train', rs=rs, rss=rss, rse=rse)
    test_set = VidPHYRECls(data_root=data_root, split='test', rs=rs, rss=rss, rse=rse)
    train_loader = DataLoader(train_set, batch_size=C.SOLVER.BATCH_SIZE // 2, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=C.SOLVER.BATCH_SIZE, shuffle=False, **kwargs)
    print(f'size: train {len(train_loader)} / test {len(test_loader)}')
    train(train_loader, test_loader, model, optim, scheduler, logger, output_dir)


if __name__ == '__main__':
    main()
