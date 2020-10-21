import phyre
import torch
import numpy as np
from glob import glob
import torch.nn.functional as F
# ---- NeuralPhys Helper Functions
from neuralphys.utils.config import _C as C
from neuralphys.utils.im import get_im_data
from neuralphys.utils.vis import plot_rollouts
from neuralphys.utils.misc import tprint, pprint
from neuralphys.utils.bbox import xyxy_to_rois, xyxy_to_posf, xywh2xyxy


class PredEvaluator(object):
    def __init__(self, device, val_loader, model, num_gpus, num_plot_image, output_dir):
        # misc
        self.device = device
        self.output_dir = output_dir
        self.num_gpus = num_gpus
        self.plot_image = num_plot_image
        # data loader
        self.val_loader = val_loader
        # nn
        self.model = model
        # input setting
        self.ptrain_size, self.ptest_size = C.RIN.PRED_SIZE_TRAIN, C.RIN.PRED_SIZE_TEST
        self.input_height, self.input_width = C.RIN.INPUT_HEIGHT, C.RIN.INPUT_WIDTH
        # loss settings
        self._setup_loss()
        self.ball_radius = 2.0  # this is not useful for now, just for compatibility
        self.high_resolution_plot = True
        # other baselines:
        self.roi_masking = C.RIN.ROI_MASKING
        self.roi_cropping = C.RIN.ROI_CROPPING
        self.vae_num_samples = 100

    def test(self):
        self.model.eval()

        if C.RIN.VAE:
            losses = dict.fromkeys(self.loss_name, 0.0)
            box_p_step_losses = [0.0 for _ in range(self.ptest_size)]
            masks_step_losses = [0.0 for _ in range(self.ptest_size)]

        for batch_idx, (data, _, rois, gt_boxes, gt_masks, valid, g_idx, _) in enumerate(self.val_loader):
            with torch.no_grad():

                if C.RIN.ROI_MASKING or C.RIN.ROI_CROPPING:
                    # data should be (b x t x o x c x h x w)
                    data = data.permute((0, 2, 1, 3, 4, 5))  # (b, o, t, c, h, w)
                    data = data.reshape((data.shape[0] * data.shape[1],) + data.shape[2:])  # (b*o, t, c, h, w)

                data = data.to(self.device)
                pos_feat = xyxy_to_posf(rois, data.shape)
                rois = xyxy_to_rois(rois, batch=data.shape[0], time_step=data.shape[1], num_devices=self.num_gpus)
                labels = {
                    'boxes': gt_boxes.to(self.device),
                    'masks': gt_masks.to(self.device),
                    'valid': valid.to(self.device),
                }
                outputs = self.model(data, rois, pos_feat, num_rollouts=self.ptest_size, g_idx=g_idx, phase='test')
                self.loss(outputs, labels, 'test')
                # VAE multiple runs
                if C.RIN.VAE:
                    vae_best_mean = np.mean(np.array(self.box_p_step_losses[:self.ptest_size]) / self.loss_cnt) * 1e3
                    losses_t = self.losses.copy()
                    box_p_step_losses_t = self.box_p_step_losses.copy()
                    masks_step_losses_t = self.masks_step_losses.copy()
                    for i in range(99):
                        outputs = self.model(data, rois, None, num_rollouts=self.ptest_size, g_idx=g_idx, phase='test')
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

                tprint(f'eval: {batch_idx}/{len(self.val_loader)}:' + ' ' * 20)

            if self.plot_image > 0:
                outputs = {
                    'boxes': outputs['boxes'].cpu().numpy(),
                    'masks': outputs['masks'].cpu().numpy() if C.RIN.MASK_LOSS_WEIGHT else None,
                }
                outputs['boxes'][..., 0::2] *= self.input_width
                outputs['boxes'][..., 1::2] *= self.input_height
                outputs['boxes'] = xywh2xyxy(
                    outputs['boxes'].reshape(-1, 4)
                ).reshape((data.shape[0], -1, C.RIN.NUM_OBJS, 4))

                labels = {
                    'boxes': labels['boxes'].cpu().numpy(),
                    'masks': labels['masks'].cpu().numpy(),
                }
                labels['boxes'][..., 0::2] *= self.input_width
                labels['boxes'][..., 1::2] *= self.input_height
                labels['boxes'] = xywh2xyxy(
                    labels['boxes'].reshape(-1, 4)
                ).reshape((data.shape[0], -1, C.RIN.NUM_OBJS, 4))

                for i in range(rois.shape[0]):
                    batch_size = C.SOLVER.BATCH_SIZE if not C.RIN.VAE else 1
                    plot_image_idx = batch_size * batch_idx + i
                    if plot_image_idx < self.plot_image:
                        tprint(f'plotting: {plot_image_idx}' + ' ' * 20)
                        video_idx, img_idx = self.val_loader.dataset.video_info[plot_image_idx]
                        video_name = self.val_loader.dataset.video_list[video_idx]

                        v = valid[i].numpy().astype(np.bool)
                        pred_boxes_i = outputs['boxes'][i][:, v]
                        gt_boxes_i = labels['boxes'][i][:, v]

                        if 'PHYRE' in C.DATA_ROOT:
                            # [::-1] is to make it consistency with others where opencv is used
                            im_data = phyre.observations_to_float_rgb(np.load(video_name).astype(np.uint8))[..., ::-1]
                            a, b, c = video_name.split('/')[5:8]
                            output_name = f'{a}_{b}_{c.replace(".npy", "")}'

                            bg_image = np.load(video_name).astype(np.uint8)
                            for fg_id in [1, 2, 3, 5]:
                                bg_image[bg_image == fg_id] = 0
                            bg_image = phyre.observations_to_float_rgb(bg_image)

                            # if f'{a}_{b}' not in [
                            #     '00014_123', '00014_528', '00015_257', '00015_337', '00019_273', '00019_296'
                            # ]:
                            #     continue

                            # if f'{a}_{b}' not in [
                            #     '00000_069', '00001_000', '00002_185', '00003_064', '00004_823',
                            #     '00005_111', '00006_033', '00007_090', '00008_177', '00009_930',
                            #     '00010_508', '00011_841', '00012_071', '00013_074', '00014_214',
                            #     '00015_016', '00016_844', '00017_129', '00018_192', '00019_244',
                            #     '00020_010', '00021_115', '00022_537', '00023_470', '00024_048'
                            # ]:
                            #     continue
                        else:
                            bg_image = None
                            image_list = sorted(glob(f'{video_name}/*{self.val_loader.dataset.image_ext}'))
                            im_name = image_list[img_idx]
                            output_name = '_'.join(im_name.split('.')[0].split('/')[-2:])
                            # deal with image data here
                            # plot rollout function only take care of the usage of plt
                            # if output_name not in ['009_015', '009_031', '009_063', '039_038', '049_011', '059_033']:
                            #     continue
                            # if output_name not in ['00002_00037', '00008_00047', '00011_00048', '00013_00036',
                            #                        '00014_00033', '00020_00054', '00021_00013', '00024_00011']:
                            #     continue
                            if output_name not in ['0016_000', '0045_000', '0120_000', '0163_000']:
                                continue
                            gt_boxes_i = labels['boxes'][i][:, v]
                            im_data = get_im_data(im_name, gt_boxes_i[None, 0:1], C.DATA_ROOT, self.high_resolution_plot)

                        if self.high_resolution_plot:
                            scale_w = im_data.shape[1] / self.input_width
                            scale_h = im_data.shape[0] / self.input_height
                            pred_boxes_i[..., [0, 2]] *= scale_w
                            pred_boxes_i[..., [1, 3]] *= scale_h
                            gt_boxes_i[..., [0, 2]] *= scale_w
                            gt_boxes_i[..., [1, 3]] *= scale_h

                        pred_masks_i = None
                        if C.RIN.MASK_LOSS_WEIGHT:
                            pred_masks_i = outputs['masks'][i][:, v]

                        plot_rollouts(im_data, pred_boxes_i, gt_boxes_i,
                                      pred_masks_i, labels['masks'][i][:, v],
                                      output_dir=self.output_dir, output_name=output_name, bg_image=bg_image)

        if C.RIN.VAE:
            self.losses = losses.copy()
            self.box_p_step_losses = box_p_step_losses.copy()
            self.loss_cnt = len(self.val_loader)

        print('\r', end='')
        print_msg = ""
        mean_loss = np.mean(np.array(self.box_p_step_losses[:self.ptest_size]) / self.loss_cnt) * 1e3
        print_msg += f"{mean_loss:.3f} | "
        print_msg += f" | ".join(["{:.3f}".format(self.losses[name] * 1e3 / self.loss_cnt) for name in self.loss_name])
        pprint(print_msg)

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
        self.losses['p_2'] = float(np.mean(self.box_p_step_losses[self.ptrain_size:])
                                   if self.ptrain_size < self.ptest_size else 0)
        self.losses['s_1'] = float(np.mean(self.box_s_step_losses[:self.ptrain_size]))
        self.losses['s_2'] = float(np.mean(self.box_s_step_losses[self.ptrain_size:])
                                   if self.ptrain_size < self.ptest_size else 0)

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
            m2_loss = self.masks_step_losses[self.ptrain_size:] if self.ptrain_size < self.ptest_size else 0
            self.losses['m_1'] = np.mean(m1_loss)
            self.losses['m_2'] = np.mean(m2_loss)
        return

    def _setup_loss(self):
        self.loss_name = []
        self.position_loss_weight = C.RIN.POSITION_LOSS_WEIGHT
        self.loss_name += ['p_1', 'p_2', 's_1', 's_2']
        if C.RIN.MASK_LOSS_WEIGHT:
            self.loss_name += ['m_1', 'm_2']
        self._init_loss()

    def _init_loss(self):
        self.losses = dict.fromkeys(self.loss_name, 0.0)
        self.box_p_step_losses = [0.0 for _ in range(self.ptest_size)]
        self.box_s_step_losses = [0.0 for _ in range(self.ptest_size)]
        self.masks_step_losses = [0.0 for _ in range(self.ptest_size)]
        # an statistics of each validation
        self.loss_cnt = 0
