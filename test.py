r"""
python test.py \
--cfg configs/phyre/pred/rpcin.yaml \
--gpus 0 \
--trans 0 \
--conv 1 \
--predictor-init outputs/phys/flash1/haozhi_baseline/ckpt_best.path.tar

Visualization:
ffmpeg -i haozhi_baseline/batch7.mp4 -i best_model/batch7.mp4 -filter_complex vstack=inputs=2 output7.mp4
"""
import os
import torch
import random
import argparse
import numpy as np

from pprint import pprint
from torch.utils.data import DataLoader

from neuralphys.models import *
from neuralphys.datasets import *
from neuralphys.utils.config import _C as C
from neuralphys.planner_phyre import PlannerPHYRE
from neuralphys.evaluator_pred import PredEvaluator


def arg_parse():
    parser = argparse.ArgumentParser(description='RPIN parameters')
    parser.add_argument('--cfg', required=True, help='path to config file', type=str)
    parser.add_argument('--predictor-init', type=str, default=None)
    parser.add_argument('--predictor-arch', type=str, default='rpcin')
    parser.add_argument('--plot-image', type=int, default=0, help='how many images are plotted')
    parser.add_argument('--gpus', type=str)
    parser.add_argument('--eval-hit', action='store_true')
    # below are only for PHYRE planning
    parser.add_argument('--start-id', default=0, type=int)
    parser.add_argument('--end-id', default=0, type=int)
    parser.add_argument('--fold-id', default=0, type=int)
    # transformer parameters
    parser.add_argument('--trans', type=int, default=0)
    parser.add_argument('--conv', type=int, default=0)
    return parser.parse_args()


def main():
    args = arg_parse()
    pprint(vars(args))
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if torch.cuda.is_available():
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        num_gpus = 1  # torch.cuda.device_count()
        print('Use {} GPUs'.format(num_gpus))
    else:
        assert NotImplementedError

    # --- setup config files
    C.merge_from_file(args.cfg)
    C.INPUT.PRELOAD_TO_MEMORY = False
    C.freeze()

    cache_name = 'figures/' + C.DATA_ROOT.split('/')[2] + '/'
    if args.predictor_init:
        cache_name += args.predictor_init.split('/')[-2]
    output_dir = os.path.join(C.OUTPUT_DIR, cache_name)

    if args.eval_hit and 'PHYRE' in C.DATA_ROOT:
        model = eval(args.predictor_arch + '.Net')(trans=args.trans, conv=args.conv)
        model.to(torch.device('cuda'))
        model = torch.nn.DataParallel(
            model, device_ids=[0]
        )
        cp = torch.load(args.predictor_init, map_location=f'cuda:0')
        model.load_state_dict(cp['model'])
        tester = PlannerPHYRE(
            device=torch.device(f'cuda'),
            num_gpus=1,
            model=model,
            output_dir=output_dir,
        )
        # tester.gen_proposal(args.start_id, args.end_id, fold_id=args.fold_id, split='train', protocal='within')
        # tester.gen_proposal(args.start_id, args.end_id, fold_id=args.fold_id, split='test', protocal='within')
        # tester.test(args.start_id, args.end_id, fold_id=args.fold_id, protocal='within')
        return

    # --- setup data loader
    print('initialize dataset')
    split_name = 'planning' if (args.eval_hit and 'PHYRE' not in C.DATA_ROOT) else 'test'
    val_set = eval(f'{C.DATASET_ABS}')(data_root=C.DATA_ROOT, split=split_name, image_ext=C.RIN.IMAGE_EXT)
    batch_size = 1 if C.RIN.VAE else C.SOLVER.BATCH_SIZE
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=0, shuffle=False)

    # prediction evaluation
    if not args.eval_hit:
        model = eval(args.predictor_arch + '.Net')(trans=args.trans, conv=args.conv)
        model.to(torch.device('cuda'))
        model = torch.nn.DataParallel(
            model, device_ids=[0]
        )
        cp = torch.load(args.predictor_init, map_location=f'cuda:0')
        model.load_state_dict(cp['model'])
        tester = PredEvaluator(
            device=torch.device('cuda'),
            val_loader=val_loader,
            num_gpus=1,
            model=model,
            num_plot_image=args.plot_image,
            output_dir=output_dir,
        )
        tester.test()
        return


if __name__ == '__main__':
    main()
