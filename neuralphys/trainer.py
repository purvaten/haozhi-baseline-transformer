import os
import torch
import numpy as np
import torch.nn.functional as F
from neuralphys.utils.misc import tprint
from timeit import default_timer as timer
from neuralphys.utils.config import _C as C
from neuralphys.utils.bbox import xyxy_to_rois, xyxy_to_posf
import pdb


class Trainer(object):
    def __init__(self, device, train_loader, val_loader, model, optim, ind,
                 max_iters, num_gpus, logger, output_dir):
        # misc
        self.device = device
        self.output_dir = output_dir
        self.logger = logger
        self.num_gpus = num_gpus
        # data loading
        self.train_loader, self.val_loader = train_loader, val_loader
        # nn optimization
        self.model = model
        self.optim = optim
        # input setting
        self.input_size = C.RIN.INPUT_SIZE
        self.ptrain_size, self.ptest_size = C.RIN.PRED_SIZE_TRAIN, C.RIN.PRED_SIZE_TEST
        self.input_height, self.input_width = C.RIN.INPUT_HEIGHT, C.RIN.INPUT_WIDTH
        self.batch_size = C.SOLVER.BATCH_SIZE
        # train loop settings
        self.iterations = 0
        self.epochs = 0
        self.max_iters = max_iters
        self.val_interval = C.SOLVER.VAL_INTERVAL
        # loss settings
        self._setup_loss()
        # timer setting
        self.best_mean = 1e6
        # indicator loss
        self.ind = ind
        # self.indicator_criterion_bce = torch.nn.BCEWithLogitsLoss(reduction='none').to(self.device)
        self.indicator_criterion_cross = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)

    def train(self):
        print_msg = "| ".join(["progress  | mean "] + list(map("{:6}".format, self.loss_name)) + list(map("{:6}".format, ['IL'])))
        self.model.train()
        print('\r', end='')
        self.logger.info(print_msg)
        while self.iterations < self.max_iters:
            self.train_epoch()
            self.epochs += 1

    def train_epoch(self):
        for batch_idx, (data, data_pred, data_t, env_name, rois, gt_boxes, gt_masks, valid, module_valid, g_idx, seq_l, objinfo, gtindicator) in enumerate(self.train_loader):
            self._adjust_learning_rate()

            if C.RIN.ROI_MASKING or C.RIN.ROI_CROPPING:
                # data should be (b x t x o x c x h x w)
                data = data.permute((0, 2, 1, 3, 4, 5))  # (b, o, t, c, h, w)
                data = data.reshape((data.shape[0] * data.shape[1],) + data.shape[2:])  # (b*o, t, c, h, w)

            data, data_t = data.to(self.device), data_t.to(self.device)
            pos_feat = xyxy_to_posf(rois, data.shape)
            rois = xyxy_to_rois(rois, batch=data.shape[0], time_step=data.shape[1], num_devices=self.num_gpus)
            self.optim.zero_grad()

            outputs = self.model(data, rois, pos_feat, valid, num_rollouts=self.ptrain_size, g_idx=g_idx, x_t=data_t, phase='train')
            labels = {
                'boxes': gt_boxes.to(self.device),
                'masks': gt_masks.to(self.device),
                'valid': valid.to(self.device),
                'module_valid': module_valid.to(self.device),
                'seq_l': seq_l.to(self.device),
                'gt_indicators': gtindicator.to(self.device),
            }
            loss = self.loss(outputs, labels, 'train')
            loss.backward()
            self.optim.step()
            # this is an approximation for printing; the dataset size may not divide the batch size
            self.iterations += self.batch_size

            print_msg = ""
            print_msg += f"{self.epochs:03}/{self.iterations // 1000:04}k"
            print_msg += f" | "
            mean_loss = np.mean(np.array(self.box_p_step_losses[:self.ptrain_size]) / self.loss_cnt) * 1e3
            print_msg += f"{mean_loss:.3f} | "
            print_msg += f" | ".join(
                ["{:.3f}".format(self.losses[name] * 1e3 / self.loss_cnt) for name in self.loss_name])
            print_msg += " || {:.4f}".format(self.loss_ind)
            speed = self.loss_cnt / (timer() - self.time)
            eta = (self.max_iters - self.iterations) / speed / 3600
            print_msg += f" | speed: {speed:.1f} | eta: {eta:.2f} h"
            print_msg += (" " * (os.get_terminal_size().columns - len(print_msg) - 10))
            tprint(print_msg)

            if self.iterations % self.val_interval == 0:
                self.snapshot()
                self.val()
                self._init_loss()
                self.model.train()

            if self.iterations >= self.max_iters:
                print('\r', end='')
                print(f'{self.best_mean:.3f}')
                break

    def val(self):
        self.model.eval()
        self._init_loss()

        if C.RIN.VAE:
            losses = dict.fromkeys(self.loss_name, 0.0)
            box_p_step_losses = [0.0 for _ in range(self.ptest_size)]
            masks_step_losses = [0.0 for _ in range(self.ptest_size)]

        for batch_idx, (data, data_pred, data_t, env_name, rois, gt_boxes, gt_masks, valid, module_valid, g_idx, seq_l, objinfo, gtindicator) in enumerate(self.val_loader):
            tprint(f'eval: {batch_idx}/{len(self.val_loader)}')
            with torch.no_grad():

                if C.RIN.ROI_MASKING or C.RIN.ROI_CROPPING:
                    data = data.permute((0, 2, 1, 3, 4, 5))
                    data = data.reshape((data.shape[0] * data.shape[1],) + data.shape[2:])

                data = data.to(self.device)
                pos_feat = xyxy_to_posf(rois, data.shape)
                rois = xyxy_to_rois(rois, batch=data.shape[0], time_step=data.shape[1], num_devices=self.num_gpus)
                labels = {
                    'boxes': gt_boxes.to(self.device),
                    'masks': gt_masks.to(self.device),
                    'valid': valid.to(self.device),
                    'module_valid': module_valid.to(self.device),
                    'seq_l': seq_l.to(self.device),
                    'gt_indicators': gtindicator.to(self.device),
                }

                outputs = self.model(data, rois, pos_feat, valid, num_rollouts=self.ptest_size, g_idx=g_idx, phase='test')
                self.loss(outputs, labels, 'test')
                # VAE multiple runs
                if C.RIN.VAE:
                    vae_best_mean = np.mean(np.array(self.box_p_step_losses[:self.ptest_size]) / self.loss_cnt) * 1e3
                    losses_t = self.losses.copy()
                    box_p_step_losses_t = self.box_p_step_losses.copy()
                    masks_step_losses_t = self.masks_step_losses.copy()
                    for i in range(9):
                        outputs = self.model(data, rois, pos_feat, valid, num_rollouts=self.ptest_size, g_idx=g_idx, phase='test')
                        self.loss(outputs, labels, 'test')
                        mean_loss = np.mean(np.array(self.box_p_step_losses[:self.ptest_size]) / self.loss_cnt) * 1e3
                        if mean_loss < vae_best_mean:
                            losses_t = self.losses.copy()
                            box_p_step_losses_t = self.box_p_step_losses.copy()
                            masks_step_losses_t = self.masks_step_losses.copy()
                            vae_best_mean = mean_loss
                        self._init_loss()

                    for k, v in losses.items():
                        losses[k] += losses_t[k]
                    for i in range(len(box_p_step_losses)):
                        box_p_step_losses[i] += box_p_step_losses_t[i]
                        masks_step_losses[i] += masks_step_losses_t[i]

        if C.RIN.VAE:
            self.losses = losses.copy()
            self.box_p_step_losses = box_p_step_losses.copy()
            self.loss_cnt = len(self.val_loader)

        print('\r', end='')
        print_msg = ""
        print_msg += f"{self.epochs:03}/{self.iterations // 1000:04}k"
        print_msg += f" | "
        mean_loss = np.mean(np.array(self.box_p_step_losses[:self.ptest_size]) / self.loss_cnt) * 1e3
        print_msg += f"{mean_loss:.3f} | "

        if mean_loss < self.best_mean:
            self.snapshot('ckpt_best.path.tar')
            self.best_mean = mean_loss

        print_msg += f" | ".join(["{:.3f}".format(self.losses[name] * 1e3 / self.loss_cnt) for name in self.loss_name])
        print_msg += " || {:.4f}".format(self.loss_ind)
        print_msg += (" " * (os.get_terminal_size().columns - len(print_msg) - 10))
        self.logger.info(print_msg)

    def loss(self, outputs, labels, phase):
        self.loss_cnt += labels['boxes'].shape[0]
        pred_size = eval(f'self.p{phase}_size')
        # calculate bbox loss
        # of shape (batch, time, #obj, 4)
        loss = (outputs['boxes'] - labels['boxes']) ** 2
        # take weighted sum over axis 2 (objs dim) since some index are not valid
        valid = labels['valid'][:, None, :, None]
        loss = loss * valid
        loss = loss.sum(2) / valid.sum(2)
        loss *= self.position_loss_weight

        for i in range(pred_size):
            self.box_p_step_losses[i] += loss[:, i, :2].sum().item()
            self.box_s_step_losses[i] += loss[:, i, 2:].sum().item()

        self.losses['p_1'] = float(np.mean(self.box_p_step_losses[:self.ptrain_size]))
        self.losses['p_2'] = float(np.mean(self.box_p_step_losses[self.ptrain_size:])) \
            if self.ptrain_size < self.ptest_size else 0
        self.losses['s_1'] = float(np.mean(self.box_s_step_losses[:self.ptrain_size]))
        self.losses['s_2'] = float(np.mean(self.box_s_step_losses[self.ptrain_size:])) \
            if self.ptrain_size < self.ptest_size else 0

        mask_loss = 0
        if C.RIN.MASK_LOSS_WEIGHT > 0:
            # of shape (batch, time, #obj, m_sz, m_sz)
            mask_loss_ = F.binary_cross_entropy(outputs['masks'], labels['masks'], reduction='none')
            mask_loss = mask_loss_.mean((3, 4))
            valid = labels['valid'][:, None, :]
            mask_loss = mask_loss * valid
            mask_loss = mask_loss.sum(2) / valid.sum(2)

            for i in range(pred_size):
                self.masks_step_losses[i] += mask_loss[:, i].sum().item()

            m1_loss = self.masks_step_losses[:self.ptrain_size]
            m2_loss = self.masks_step_losses[self.ptrain_size:]
            self.losses['m_1'] = np.mean(m1_loss)
            self.losses['m_2'] = np.mean(m2_loss) if self.ptrain_size < self.ptest_size else 0

            mask_loss = mask_loss.mean(0)
            init_tau = C.RIN.DISCOUNT_TAU ** (1 / self.ptrain_size)
            tau = init_tau + (self.iterations / self.max_iters) * (1 - init_tau)
            tau = torch.pow(tau, torch.arange(pred_size, out=torch.FloatTensor()))[:, None].to('cuda')
            mask_loss = ((mask_loss * tau) / tau.sum(axis=0, keepdims=True)).sum()
            mask_loss = mask_loss * C.RIN.MASK_LOSS_WEIGHT

        seq_loss = 0
        if C.RIN.SEQ_CLS_LOSS_WEIGHT > 0:
            seq_loss = F.binary_cross_entropy(outputs['score'], labels['seq_l'], reduction='none')
            self.losses['seq'] += seq_loss.sum().item()
            seq_loss = seq_loss.mean() * C.RIN.SEQ_CLS_LOSS_WEIGHT

        kl_loss = 0
        if C.RIN.VAE and phase == 'train':
            kl_loss = outputs['kl']
            self.losses['kl'] += kl_loss.sum().item()
            kl_loss = C.RIN.VAE_KL_LOSS_WEIGHT * kl_loss.sum()

        # no need to do precise batch statistics, just do mean for backward gradient
        loss = loss.mean(0)
        init_tau = C.RIN.DISCOUNT_TAU ** (1 / self.ptrain_size)
        tau = init_tau + (self.iterations / self.max_iters) * (1 - init_tau)
        tau = torch.pow(tau, torch.arange(pred_size, out=torch.FloatTensor()))[:, None].to('cuda')
        loss = ((loss * tau) / tau.sum(axis=0, keepdims=True)).sum()
        loss = loss + mask_loss + kl_loss + seq_loss

        # **************************************************************************************************************************** #
        # INDICATOR LOSS
        # **************************************************************************************************************************** #
        gt, pred = labels['gt_indicators'], outputs['pred_indicators']

        # # bce loss
        # valid1 = labels['valid'][:, None, :, None]    # (b, 1, 6, 1)
        # valid2 = labels['valid'][:, None, None, :]    # (b, 1, 1, 6)
        # ind_loss = self.indicator_criterion_bce(pred, gt)    # (b, 5, 6, 6)
        # ind_loss = ind_loss * valid1 * valid2    # mask out invalid rows & columns
        # ind_loss = ind_loss.sum(2) / valid1.sum(2)        # loss is mean of all valid objects (b, 5)

        # cross entropy loss
        valid = labels['valid'][:, None, :]    # (b, 1, 6)
        ind_loss = self.indicator_criterion_cross(pred.reshape(-1, 6), torch.argmax(gt, 3).flatten()).reshape(-1, pred_size, gt.shape[2])    # (b, 5, 6)
        ind_loss = ind_loss * valid    # mask out invalid rows (objects)
        ind_loss = ind_loss.sum(2) / valid.sum(2)        # loss is mean of all valid objects (b, 5)

        self.loss_ind = ((ind_loss.mean(0) * tau) / tau.sum(axis=0, keepdims=True)).sum()
        loss += self.ind * self.loss_ind
        # **************************************************************************************************************************** #

        return loss

    def snapshot(self, name='ckpt_latest.path.tar'):
        torch.save(
            {
                'arch': self.model.__class__.__name__,
                'model': self.model.state_dict(),
                'optim': self.optim.state_dict(),
                'iterations': self.iterations,
                'epochs': self.epochs,
                'best_mean': self.best_mean,
            },
            os.path.join(self.output_dir, name),
        )

    def _setup_loss(self):
        self.loss_name = []
        self.position_loss_weight = C.RIN.POSITION_LOSS_WEIGHT
        self.loss_name += ['p_1', 'p_2', 's_1', 's_2']
        if C.RIN.MASK_LOSS_WEIGHT:
            self.loss_name += ['m_1', 'm_2']
        if C.RIN.VAE:
            self.loss_name += ['kl']
        if C.RIN.SEQ_CLS_LOSS_WEIGHT:
            self.loss_name += ['seq']
        self._init_loss()

    def _init_loss(self):
        self.losses = dict.fromkeys(self.loss_name, 0.0)
        self.box_p_step_losses = [0.0 for _ in range(self.ptest_size)]
        self.box_s_step_losses = [0.0 for _ in range(self.ptest_size)]
        self.masks_step_losses = [0.0 for _ in range(self.ptest_size)]
        # an statistics of each validation
        self.loss_cnt = 0
        self.loss_ind = 0
        self.time = timer()

    def _adjust_learning_rate(self):
        if self.iterations <= C.SOLVER.WARMUP_ITERS:
            lr = C.SOLVER.BASE_LR * self.iterations / C.SOLVER.WARMUP_ITERS
        else:
            if C.SOLVER.SCHEDULER == 'step':
                lr = C.SOLVER.BASE_LR
                for m_iters in C.SOLVER.LR_MILESTONES:
                    if self.iterations > m_iters:
                        lr *= C.SOLVER.LR_GAMMA
            elif C.SOLVER.SCHEDULER == 'cosine':
                lr = 0.5 * C.SOLVER.BASE_LR * (1 + np.cos(np.pi * self.iterations / self.max_iters))
            else:
                raise NotImplementedError

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
