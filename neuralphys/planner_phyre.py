import os
import cv2
import torch
import phyre
import hickle
import pickle
import random
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from neuralphys.models.dqn import ResNet18Film, ResNet18FilmAction
from neuralphys.utils.config import _C as C
from neuralphys.utils.misc import tprint, pprint
from neuralphys.utils.bbox import xyxy_to_rois, xyxy_to_posf, xywh2xyxy


class PlannerPHYRE(object):
    def __init__(self, device, model, num_gpus, output_dir):
        self.device = device
        self.output_dir = output_dir
        self.num_gpus = num_gpus
        self.model = model
        self.input_size = C.RIN.INPUT_SIZE
        self.input_height, self.input_width = C.RIN.INPUT_HEIGHT, C.RIN.INPUT_WIDTH
        self.pred_rollout = C.RIN.PRED_SIZE_TEST
        self.model.eval()

        self.score_with_heuristic = False  # the most naive method
        self.score_with_act = False  # image + action
        self.score_with_mask = True  # image + predicted mask
        self.score_with_vid_cls = False  # gt image, for upper bound test

        if self.score_with_heuristic:
            self.method = 'heuristic'
            self.score_model = None
            self.score_model_cache = None
        else:
            self.score_model_cache = 'outputs/phys/cls'
            if self.score_with_act:
                # deprecated
                # self.score_model_cache = None
                # self.score_model = ResNet18FilmAction(3, fusion_place='last', action_hidden_size=256,
                #                                       action_layers=1).to(torch.device('cuda'))
                raise NotImplementedError
            elif self.score_with_vid_cls or self.score_with_mask:
                self.method = 'gt0+pred3+pred6'
                self.proposal_setting = 'p200n300a100'
                self.rendered_mask_idx = [
                    [2, 5],
                    # [0, 2, 4],
                    # [1, 3, 5],
                    # [2, 4, 6],
                    # [3, 5, 7],
                ]
                # self.gt_image_idx = [0, 2, 4, 6]
                self.score_model_cache = f'{self.score_model_cache}/' \
                                         f'w0_rpcin_0_t10_d256_lr2e4_mlw0003_{self.proposal_setting}/' \
                                         f'{self.method}/last.ckpt'
                self.score_model = ResNet18Film(3, 'early').to(torch.device('cuda'))
            else:
                raise NotImplementedError

            cp = torch.load(self.score_model_cache, map_location=f'cuda:0')
            self.score_model.load_state_dict(cp['model'])
            self.score_model.eval()
            print(self.score_model_cache)

    def test(self, start_id=0, end_id=25, fold_id=0, protocal='within'):
        random.seed(0)
        print(f'testing {protocal} fold {fold_id}')
        eval_setup = f'ball_{protocal}_template'
        action_tier = phyre.eval_setup_to_action_tier(eval_setup)
        _, _, test_tasks = phyre.get_fold(eval_setup, fold_id)  # PHYRE setup
        candidate_list = [f'{i:05d}' for i in range(start_id, end_id)]  # filter tasks
        test_list = [task for task in test_tasks if task.split(':')[0] in candidate_list]
        simulator = phyre.initialize_simulator(test_list, action_tier)
        # PHYRE evaluation
        num_all_actions = [1000, 2000, 5000, 8000, 10000]
        auccess = np.zeros((len(num_all_actions), len(test_list), 100))
        batched_pred = C.SOLVER.BATCH_SIZE
        # DATA for network:
        all_data, all_acts, all_rois, all_image = [], [], [], []
        cache = phyre.get_default_100k_cache('ball')
        acts = cache.action_array[:10000]
        # actions = cache.action_array[:100000]
        # training_data = cache.get_sample(test_list, None)

        pos_all, neg_all, pos_correct, neg_correct = 0, 0, 0, 0

        objs_color = None
        for task_id, task in enumerate(test_list):
            confs, successes, num_valid_act_idx = [], [], []

            boxes_cache_name = f'cache/{task.replace(":", "_")}.hkl'
            use_cache = os.path.exists(boxes_cache_name)
            all_boxes = hickle.load(boxes_cache_name) if use_cache else []

            valid_act_cnt = 0

            # sim_statuses = training_data['simulation_statuses'][task_id]
            # pos_acts = actions[sim_statuses == 1]
            # neg_acts = actions[sim_statuses == -1]
            # np.random.shuffle(pos_acts)
            # np.random.shuffle(neg_acts)
            # pos_acts = pos_acts[:50]
            # neg_acts = neg_acts[:200]
            # acts = np.concatenate([pos_acts, neg_acts])

            for act_id, act in enumerate(acts):
                if act_id == 0:
                    pprint(f'{task}: {task_id} / {len(test_list)}')
                sim = simulator.simulate_action(task_id, act, stride=60, need_images=True, need_featurized_objects=True)
                if sim.status == phyre.SimulationStatus.INVALID_INPUT:
                    num_valid_act_idx.append(0)
                    if act_id == len(acts) - 1 and len(all_data) > 0:  # final action is invalid
                        conf_t = self.batch_score(all_data, all_acts, all_rois, all_image, objs_color, task)
                        confs = confs + conf_t
                        all_data, all_acts, all_rois, all_image = [], [], [], []
                    continue
                num_valid_act_idx.append(1)
                successes.append(sim.status == phyre.SimulationStatus.SOLVED)

                if self.score_with_heuristic or self.score_with_mask:
                    # parse object, prepare input for network:
                    image = cv2.resize(sim.images[0], (self.input_width, self.input_height),
                                       interpolation=cv2.INTER_NEAREST)
                    all_image.append(image[::-1])  # for heuristic method to detect goal location, need to flip
                    image = phyre.observations_to_float_rgb(image)
                    objs_color = sim.featurized_objects.colors
                    objs_valid = [('BLACK' not in obj_color) and ('PURPLE' not in obj_color) for obj_color in objs_color]
                    objs = sim.featurized_objects.features[:, objs_valid, :]
                    objs_color = np.array(objs_color)[objs_valid]
                    num_objs = objs.shape[1]

                    if use_cache:
                        boxes = all_boxes[valid_act_cnt]
                        valid_act_cnt += 1
                    else:
                        boxes = np.zeros((1, num_objs, 5))
                        for o_id in range(num_objs):
                            mask = phyre.objects_util.featurized_objects_vector_to_raster(objs[0][[o_id]])
                            mask_im = phyre.observations_to_float_rgb(mask)
                            mask_im[mask_im == 1] = 0
                            mask_im = mask_im.sum(-1) > 0

                            [h, w] = np.where(mask_im)
                            x1, x2, y1, y2 = w.min(), w.max(), h.min(), h.max()
                            x1 *= (self.input_width - 1) / (phyre.SCENE_WIDTH - 1)
                            x2 *= (self.input_width - 1) / (phyre.SCENE_WIDTH - 1)
                            y1 *= (self.input_height - 1) / (phyre.SCENE_HEIGHT - 1)
                            y2 *= (self.input_height - 1) / (phyre.SCENE_HEIGHT - 1)
                            boxes[0, o_id] = [o_id, x1, y1, x2, y2]
                        all_boxes.append(boxes)

                    data = image.transpose((2, 0, 1))[None, None, :]
                    data = torch.from_numpy(data.astype(np.float32))
                    rois = torch.from_numpy(boxes[..., 1:].astype(np.float32))[None, :]

                    all_data.append(data)
                    all_rois.append(rois)
                elif self.score_with_act:
                    init = np.ascontiguousarray(simulator.initial_scenes[task_id][::-1])
                    init128 = cv2.resize(init, (self.input_width, self.input_height),
                                         interpolation=cv2.INTER_NEAREST)
                    all_data.append(torch.from_numpy(init128))
                    all_acts.append(torch.from_numpy(act[None, :]))
                elif self.score_with_vid_cls:
                    rst_images = np.stack([np.ascontiguousarray(
                        cv2.resize(rst_image, (self.input_width, self.input_height),
                                   interpolation=cv2.INTER_NEAREST)[::-1]
                    ) for rst_image in sim.images])
                    all_data.append(torch.from_numpy(rst_images))
                else:
                    raise NotImplementedError

                if len(all_data) % batched_pred == 0 or act_id == len(acts) - 1:
                    conf_t = self.batch_score(all_data, all_acts, all_rois, all_image, objs_color, task)
                    confs = confs + conf_t
                    all_data, all_acts, all_rois, all_image = [], [], [], []

            if self.score_with_heuristic or self.score_with_mask:
                if not use_cache:
                    all_boxes = np.stack(all_boxes)
                    hickle.dump(all_boxes, boxes_cache_name, mode='w', compression='gzip')
                else:
                    assert valid_act_cnt == len(all_boxes)

            pred = np.array(confs) >= 0.5
            labels = np.array(successes)

            pos_all += (labels == 1).sum()
            neg_all += (labels == 0).sum()
            pos_correct += (pred == labels)[labels == 1].sum()
            neg_correct += (pred == labels)[labels == 0].sum()

            pos_acc = (pred == labels)[labels == 1].sum() / (labels == 1).sum()
            neg_acc = (pred == labels)[labels == 0].sum() / (labels == 0).sum()
            info = f'{pos_acc * 100:.1f} / {neg_acc * 100:.1f} '
            # info = f'{task}: '
            for j, num_acts in enumerate(num_all_actions):
                num_valid = np.sum(num_valid_act_idx[:num_acts])
                top_acc = np.array(successes[:num_valid])[np.argsort(confs[:num_valid])[::-1]]
                for i in range(100):
                    auccess[j, task_id, i] = int(np.sum(top_acc[:i + 1]) > 0)
                w = np.array([np.log(k + 1) - np.log(k) for k in range(1, 101)])
                s = auccess[j, :task_id + 1].sum(0) / auccess[j, :task_id + 1].shape[0]
                info += f'{np.sum(w * s) / np.sum(w) * 100:.2f} {np.sum(successes[:num_valid])}/{num_acts // 1000}k | '
            pprint(info)
        pprint(pos_correct, pos_all, pos_correct / pos_all)
        pprint(neg_correct, neg_all, neg_correct / neg_all)
        cache_output_dir = f'{self.output_dir.replace("figures/", "")}/' \
                           f'{self.proposal_setting}_{self.method}_{protocal}_fold_{fold_id}/'
        os.makedirs(cache_output_dir, exist_ok=True)
        print(cache_output_dir)
        stats = {
            'auccess': auccess,
            'p_c': pos_correct,
            'p_a': pos_all,
            'n_c': neg_correct,
            'n_a': neg_all,
        }
        with open(f'{cache_output_dir}/{start_id}_{end_id}.pkl', 'wb') as f:
            pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)

    def gen_proposal(self, start_id=0, end_id=25, fold_id=0, split='train', protocal='within'):
        max_p_acts, max_n_acts, max_acts = 50, 200, 100000
        self.proposal_dir = f'{protocal[0]}{fold_id}_{self.output_dir.split("/")[-1]}_' \
                            f'p{max_p_acts}n{max_n_acts}a{max_acts // 1000}'
        eval_setup = f'ball_{protocal}_template'
        action_tier = phyre.eval_setup_to_action_tier(eval_setup)
        train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
        # filter task
        train_tasks = train_tasks + dev_tasks
        candidate_list = [f'{i:05d}' for i in range(start_id, end_id)]
        train_list = [task for task in train_tasks if task.split(':')[0] in candidate_list]
        test_list = [task for task in test_tasks if task.split(':')[0] in candidate_list]
        if len(eval(f'{split}_list')) == 0:
            return

        simulator = phyre.initialize_simulator(eval(f'{split}_list'), action_tier)
        cache = phyre.get_default_100k_cache('ball')
        training_data = cache.get_sample(eval(f'{split}_list'), None)
        actions = cache.action_array[:max_acts]

        final_list = eval(f'{split}_list')
        for task_id, task in enumerate(final_list):
            pprint(f'{task}: {task_id} / {len(final_list)}')
            sim_statuses = training_data['simulation_statuses'][task_id]
            pos_acts = actions[sim_statuses == 1]
            neg_acts = actions[sim_statuses == -1]
            np.random.shuffle(pos_acts)
            np.random.shuffle(neg_acts)
            pos_acts = pos_acts[:max_p_acts]
            neg_acts = neg_acts[:max_n_acts]
            acts = np.concatenate([pos_acts, neg_acts])

            for act_id, act in enumerate(acts):
                sim = simulator.simulate_action(task_id, act, stride=60, need_images=True, need_featurized_objects=True)
                assert sim.status != phyre.SimulationStatus.INVALID_INPUT
                raw_images = sim.images

                rst_images = np.stack([np.ascontiguousarray(
                    cv2.resize(rst_image, (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)[::-1]
                ) for rst_image in raw_images])

                # prepare input for network:
                image = cv2.resize(raw_images[0], (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)
                image = phyre.observations_to_float_rgb(image)
                # parse object
                objs_color = sim.featurized_objects.colors
                objs_valid = [('BLACK' not in obj_color) and ('PURPLE' not in obj_color) for obj_color in objs_color]
                objs = sim.featurized_objects.features[:, objs_valid, :]
                objs_color = np.array(objs_color)[objs_valid]
                num_objs = objs.shape[1]
                boxes = np.zeros((1, num_objs, 5))
                for o_id in range(num_objs):
                    mask = phyre.objects_util.featurized_objects_vector_to_raster(objs[0][[o_id]])
                    mask_im = phyre.observations_to_float_rgb(mask)
                    mask_im[mask_im == 1] = 0
                    mask_im = mask_im.sum(-1) > 0

                    [h, w] = np.where(mask_im)
                    x1, x2, y1, y2 = w.min(), w.max(), h.min(), h.max()
                    x1 *= (self.input_width - 1) / (phyre.SCENE_WIDTH - 1)
                    x2 *= (self.input_width - 1) / (phyre.SCENE_WIDTH - 1)
                    y1 *= (self.input_height - 1) / (phyre.SCENE_HEIGHT - 1)
                    y2 *= (self.input_height - 1) / (phyre.SCENE_HEIGHT - 1)
                    boxes[0, o_id] = [o_id, x1, y1, x2, y2]

                data = image.transpose((2, 0, 1))[None, None, :]
                data = torch.from_numpy(data.astype(np.float32))
                rois = torch.from_numpy(boxes[..., 1:].astype(np.float32))[None, :]

                bg_image = rst_images[0].copy()
                for fg_id in [1, 2, 3, 5]:
                    bg_image[bg_image == fg_id] = 0
                boxes, masks = self.generate_trajs(data, rois)
                rst_masks = np.stack([
                    self.render_mask_to_image(boxes[0, i], masks[0, i], images=bg_image.copy(), color=objs_color)
                        .astype(np.uint8) for i in range(self.pred_rollout)
                ])

                output_dir = f'data/dynamics/{self.proposal_dir}/{split}/'
                output_dir = output_dir + 'pos/' if sim.status == phyre.SimulationStatus.SOLVED else output_dir + 'neg/'
                output_dir = output_dir + f'{task.replace(":", "_")}/'
                os.makedirs(output_dir, exist_ok=True)
                rst_dict = {'gt_im': rst_images, 'pred_im': rst_masks}
                hickle.dump(rst_dict, f'{output_dir}/{act_id}.hkl', mode='w', compression='gzip')

                # # render mask to image
                # # debug images:
                # debug_dir = f'data/dynamics/{self.proposal_dir}/{split}/debug/'
                # debug_dir = debug_dir + f'{task.replace(":", "_")}/'
                # os.makedirs(debug_dir, exist_ok=True)
                #
                # plt.figure(figsize=(12, 12))
                # plt.subplot(2, 2, 1)
                # plt.imshow(phyre.observations_to_float_rgb(rst_images[1])[::-1])
                # plt.subplot(2, 2, 2)
                # final_image = rst_images[8] if len(rst_images) >= 9 else rst_images[-1]
                # plt.imshow(phyre.observations_to_float_rgb(final_image)[::-1])
                # plt.subplot(2, 2, 3)
                # plt.imshow(phyre.observations_to_float_rgb(rst_masks[0])[::-1])
                # plt.subplot(2, 2, 4)
                # plt.imshow(phyre.observations_to_float_rgb(rst_masks[7])[::-1])
                # plt.savefig(f'{debug_dir}/{act_id}.jpg'), plt.close()

    def render_mask_to_image(self, boxes, masks, images=None, color=None):
        if images is None:
            images = np.zeros((self.input_height, self.input_width))

        color_dict = {
            'WHITE': 0, 'RED': 1, 'GREEN': 2, 'BLUE': 3, 'PURPLE': 4, 'GRAY': 5, 'BLACK': 6
        }

        for o_id, (box, mask) in enumerate(zip(boxes, masks)):
            assert self.input_width == self.input_height
            box = np.maximum(np.minimum(np.round(box).astype(np.int), self.input_height - 1), 0)
            if box[2] - box[0] + 1 <= 0 or box[3] - box[1] + 1 <= 0:
                continue
            mask = cv2.resize(mask, (box[2] - box[0] + 1, box[3] - box[1] + 1))
            mask = (mask >= 0.5)
            images[box[1]:box[3] + 1, box[0]:box[2] + 1][mask] = color_dict[color[o_id]]
        return images

    def get_act_conf(self, traj_array, task, image, color_list):
        # green thing touch blue thing
        cat1 = ['00000', '00004', '00006', '00018', '00022', '00023']
        # green thing touch purple bar (x-direction)
        cat2 = ['00001', '00003', '00010']
        # green thing touch purple bar (y-direction)
        cat3 = ['00002', '00005', '00009', '00012', '00013', '00015', '00017', '00024']
        # green thing touch purple bar (x,y-direction)
        cat4 = ['00007', '00008', '00011', '00014', '00016', '00019', '00020', '00021']

        [h, w] = np.where(image == 4)
        if len(h) == 0 or len(w) == 0:
            goal_x = goal_y = 0
        else:
            x1, x2, y1, y2 = w.min(), w.max(), h.min(), h.max()
            goal_x = 0.5 * (x1 + x2)
            goal_y = 0.5 * (y1 + y2)

        # the order is green, blue, red
        green_id = np.where(color_list == 'GREEN')[0].item()
        green_x = (traj_array[-1, green_id, 0] + traj_array[-1, green_id, 2]) / 2
        green_y = (traj_array[-1, green_id, 1] + traj_array[-1, green_id, 3]) / 2
        if task.split(':')[0] in cat1:
            blue_id = np.where(color_list == 'BLUE')[0].item()
            blue_x = (traj_array[-1, blue_id, 0] + traj_array[-1, blue_id, 2]) / 2
            blue_y = (traj_array[-1, blue_id, 1] + traj_array[-1, blue_id, 3]) / 2
            conf = -np.sqrt((green_x - blue_x) ** 2 + (green_y - blue_y) ** 2)
        elif task.split(':')[0] in cat2:
            conf = -np.sqrt((green_x - goal_x) ** 2)
        elif task.split(':')[0] in cat3:
            conf = -np.sqrt((green_y - goal_y) ** 2)
        elif task.split(':')[0] in cat4:
            conf = -np.sqrt(((green_x - goal_x) ** 2 + (green_y - goal_y) ** 2)).min()
        else:
            raise NotImplementedError
        return conf

    @staticmethod
    def enumerate_actions():
        tier = 'ball'
        actions = phyre.get_default_100k_cache(tier).action_array[:10000]
        return actions

    def batch_score(self, all_data, all_acts=None, all_rois=None, all_image=None, objs_color=None, task=None):
        if self.score_with_heuristic:
            all_data = torch.cat(all_data)
            all_rois = torch.cat(all_rois)
            boxes, masks = self.generate_trajs(all_data, all_rois)
            confs = [self.get_act_conf(boxes[i], task, all_image[i], objs_color) for i in range(boxes.shape[0])]
        elif self.score_with_mask:
            # ours models
            all_data = torch.cat(all_data)
            all_rois = torch.cat(all_rois)
            boxes, masks = self.generate_trajs(all_data, all_rois)

            confs = 0

            for d_i, rendered_mask_idx in enumerate(self.rendered_mask_idx):
                rendered_masks = [[] for _ in rendered_mask_idx]
                for i in range(boxes.shape[0]):
                    bg_image = all_image[i].copy()
                    for fg_id in [1, 2, 3, 5]:
                        bg_image[bg_image == fg_id] = 0

                    for j, mask_idx in enumerate(rendered_mask_idx):
                        rendered_masks[j].append(self.render_mask_to_image(
                            boxes[i, mask_idx], masks[i, mask_idx], images=bg_image.copy(),  color=objs_color
                        ).astype(np.uint8))

                if d_i == 0:
                    data = [torch.from_numpy(np.stack(all_image))] \
                           + [torch.from_numpy(np.stack(env_mask)) for env_mask in rendered_masks]
                else:
                    data = [torch.from_numpy(np.stack(env_mask)) for env_mask in rendered_masks]
                data = torch.cat([d[:, None] for d in data], dim=1)

                with torch.no_grad():
                    confs = confs + self.score_model(data).sigmoid().cpu().numpy()
            confs /= len(self.rendered_mask_idx)
        elif self.score_with_vid_cls:
            # DQN models
            gt_images = []
            for i, im_idx in enumerate(self.gt_image_idx):
                gt_images.append(torch.stack([c[im_idx] if len(c) >= im_idx + 1 else c[-1] for c in all_data]))
            with torch.no_grad():
                if self.score_with_act:
                    raise NotImplementedError
                    # input_acts = torch.cat(all_acts)
                    # confs = self.score_model(data, input_acts).sigmoid().cpu().numpy()
                else:
                    data = torch.cat([d[:, None] for d in gt_images], dim=1)
                    confs = self.score_model(data).sigmoid().cpu().numpy()
        else:
            raise NotImplementedError

        return list(confs)

    def generate_trajs(self, data, boxes):
        with torch.no_grad():
            num_objs = boxes.shape[2]
            g_idx = np.array([[i, j, 1] for i in range(num_objs) for j in range(num_objs) if j != i])
            g_idx = torch.from_numpy(g_idx[None].repeat(data.shape[0], 0))
            rois = xyxy_to_rois(boxes, batch=data.shape[0], time_step=data.shape[1], num_devices=self.num_gpus)
            pos_feat = xyxy_to_posf(rois, data.shape)
            outputs = self.model(data, rois, pos_feat, num_rollouts=self.pred_rollout, g_idx=g_idx)
            outputs = {
                'boxes': outputs['boxes'].cpu().numpy(),
                'masks': outputs['masks'].cpu().numpy(),
            }
            outputs['boxes'][..., 0::2] *= self.input_width
            outputs['boxes'][..., 1::2] *= self.input_height
            outputs['boxes'] = xywh2xyxy(
                outputs['boxes'].reshape(-1, 4)
            ).reshape((data.shape[0], -1, num_objs, 4))

        return outputs['boxes'], outputs['masks']
