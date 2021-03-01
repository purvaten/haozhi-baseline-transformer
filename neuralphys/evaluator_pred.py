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
import matplotlib.pyplot as plt
import subprocess
import pdb
import os
import cv2


def _hex_to_ints(hex_string):
    hex_string = hex_string.strip('#')
    return (
        int(hex_string[0:2], 16),
        int(hex_string[2:4], 16),
        int(hex_string[4:6], 16),
    )


WAD_COLORS = np.array(
    [
        [255, 255, 255],  # White.
        _hex_to_ints('f34f46'),  # Red.
        _hex_to_ints('6bcebb'),  # Green.
        _hex_to_ints('1877f2'),  # Blue.
        _hex_to_ints('4b4aa4'),  # Purple.
        _hex_to_ints('b9cad2'),  # Gray.
        [0, 0, 0],  # Black.
        _hex_to_ints('fcdfe3'),  # Light red.
    ],
    dtype=np.uint8)


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

        for batch_idx, (data, data_pred, data_t, env_name, rois, gt_boxes, gt_masks, valid, module_valid, g_idx, _) in enumerate(self.val_loader):
            with torch.no_grad():

                # decide module_valid here for evaluation
                mid = 0    # ball-only
                module_valid = module_valid[:, mid, :, :]

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
                    'module_valid': module_valid.to(self.device),
                }
                outputs = self.model(data, rois, pos_feat, num_rollouts=self.ptest_size, g_idx=g_idx, phase='test')

                # *********************************************************************************
                # VISUALIZATION - generate input image and GT outputs + model outputs
                input_data = data.cpu().detach().numpy()
                gt_data = data_pred.cpu().detach().numpy()
                data_t = data_t.cpu().detach().numpy()
                validity = valid.cpu().detach().numpy()
                self.visualize_results(input_data, gt_data, outputs, data_t, validity, env_name)
                # *********************************************************************************

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
        valid = labels['valid'][:, None, :, None]    # use this for base loss
        # valid = labels['module_valid'][:, :, :, None]    # use this for module specific loss
        valid += 1e-15
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
            valid = labels['valid'][:, None, :]    # use this for base loss
            # valid = labels['module_valid']    # use this for module specific loss
            valid += 1e-15
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

    def visualize_results(self, input_data, gt_data, outputs, data_t, valid, env_names):
        """Display input data, GT results and predicted results."""
        # generate background images for this batch based on first frame
        bg_imgs = data_t[:, 0, :, :]
        for fg_id in [1, 2, 3, 5]:
            bg_imgs[bg_imgs == fg_id] = 0
        bgrounds = np.array([phyre.observations_to_float_rgb(bg_img.astype(int)) for bg_img in bg_imgs], dtype=np.float).transpose((0, 3, 1, 2))

        # adjust the output boxes
        outputs['boxes'], outputs['masks'] = outputs['boxes'].cpu().detach().numpy(), outputs['masks'].cpu().detach().numpy()
        outputs['boxes'][..., 0::2] *= self.input_width
        outputs['boxes'][..., 1::2] *= self.input_height
        outputs['boxes'] = xywh2xyxy(
            outputs['boxes'].reshape(-1, 4)
        ).reshape((input_data.shape[0], -1, C.RIN.NUM_OBJS, 4))

        # get number of objects
        num_objs = np.sum(valid == 1, 1)

        for b in range(input_data.shape[0]):
            # # display input
            # plt.imshow(input_data[b][0].transpose(1,2,0))
            # plt.savefig('images/input' + str(b) + '.png')

            # delete older images in the folders
            print("Starting file deletions")
            for item in os.listdir('images/gt/'): os.remove(os.path.join('images/gt/', item))
            for item in os.listdir('images/pred/'): os.remove(os.path.join('images/pred/', item))
            print("File deletions complete")

            # display GT results
            for t in range(gt_data.shape[1]):
                plt.title('GT')
                plt.text(20, 5, 'time: '+str(t), horizontalalignment='center', verticalalignment='center')
                plt.imshow(gt_data[b][t].transpose(1,2,0))
                plt.savefig('images/gt/' + str(t) + '.png')
                plt.close()

            # display model predictions
            for t in range(outputs['boxes'].shape[1]):
                bbox = outputs['boxes'][b, t, :, :]
                mask = outputs['masks'][b, t, :, :, :]
                env_name = env_names[b]
                bground, num_obj = bgrounds[b], num_objs[b]
                pred_img = self.img_from_object_feature(bground, bbox, mask, num_obj, env_name)
                plt.title('Prediction')
                plt.text(20, 5, 'time: '+str(t), horizontalalignment='center', verticalalignment='center')
                plt.imshow(pred_img.transpose(1,2,0))
                plt.savefig('images/pred/' + str(t) + '.png')
                plt.close()

            # generate video of this batch
            pdb.set_trace()
            self.save_video('images/pred/', b)
            self.save_video('images/gt/', b)

            # generate videos side by side
            p = subprocess.Popen(['ffmpeg', '-i', 'gt/batch'+str(b)+'.mp4', '-i', 'pred/batch'+str(b)+'.mp4', '-filter_complex', '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]', '-map', '[vid]', '-c:v', 'libx264', '-crf', '23', '-preset', 'veryfast', 'batch'+str(b)+'.mp4'], cwd='images/')
            p.wait()

    
    def save_video(self, foldername, batch_id):
        """Make video from given frames."""
        p = subprocess.Popen(['ffmpeg', '-f', 'image2', '-r', '1', '-i', '%d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', 'batch'+str(batch_id)+'.mp4'], cwd=foldername)
        p.wait()


    def img_from_object_feature(self, bg_img, bbox, mask, num_obj, env_name):
        """Generate new image given
            bg_img      (3, 128, 128)
            bboxes      (6, 4)
            mask        (6, 21, 21)
            num_obj     int
            env_names   str
        """
        pred_mask_im = bg_img.copy() * 255

        # decide mask color
        if env_name in ['00000']:
            mask_colors = [WAD_COLORS[2], WAD_COLORS[3], WAD_COLORS[1]]
        elif env_name in ['00004', '00006']:  # BLUE + GREEN
            mask_colors = [WAD_COLORS[3], WAD_COLORS[2], WAD_COLORS[1]]
        elif env_name in ['00001', '00002', '00007', '00008', '00009', '00011', '00012', '00013', '00014', '00015']:
            mask_colors = [WAD_COLORS[2], WAD_COLORS[1]]
        elif env_name in ['00003', '00005', '00010', '00017', '00021']:  # GRAY + GREEN
            mask_colors = [WAD_COLORS[5], WAD_COLORS[2], WAD_COLORS[1]]
        elif env_name in ['00018']:  # 2 GRAY + GREEN + BLUE
            mask_colors = [WAD_COLORS[5], WAD_COLORS[3], WAD_COLORS[5], WAD_COLORS[2], WAD_COLORS[1]]
        elif env_name in ['00022']:  # 1 GRAY + GREEN + BLUE
            mask_colors = [WAD_COLORS[5], WAD_COLORS[3], WAD_COLORS[2], WAD_COLORS[1]]
        elif env_name in ['00023']:  # 1 GRAY + GREEN + BLUE
            mask_colors = [WAD_COLORS[5], WAD_COLORS[5], WAD_COLORS[2], WAD_COLORS[5], WAD_COLORS[3], WAD_COLORS[1]]
        elif env_name in ['00016', '00019', '00020', '00024']:
            mask_colors = [WAD_COLORS[2], WAD_COLORS[5], WAD_COLORS[1]]

        # plot each dynamic object
        for o_id in range(num_obj):
            # resize mask to bbox size
            pred_bbox_t_o = np.maximum(np.minimum(np.round(bbox[o_id]).astype(np.int), 127), 0)
            pred_mask_t_o = cv2.resize(mask[o_id], (pred_bbox_t_o[2] - pred_bbox_t_o[0] + 1, pred_bbox_t_o[3] - pred_bbox_t_o[1] + 1))
            pred_mask_t_o = (pred_mask_t_o >= 0.5)

            for c_id in range(3):
                # for 3 channels
                pred_mask_im[c_id, pred_bbox_t_o[1]:pred_bbox_t_o[3] + 1,
                                        pred_bbox_t_o[0]:pred_bbox_t_o[2] + 1][pred_mask_t_o] = mask_colors[o_id][c_id]

        return pred_mask_im / 255.
