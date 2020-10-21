# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This library contains actual implementation of the DQN agent."""
from typing import Optional, Sequence, Tuple
import glob
import argparse
import logging
import os
import time

import numpy as np
import torch

from neuralphys.models import dqn as nets
from neuralphys.utils.misc import tprint
from neuralphys.datasets.cls import PHYRECls
import phyre

AUCCESS_EVAL_TASKS = 100000
XE_EVAL_SIZE = 10000

TaskIds = Sequence[str]
NeuralModel = torch.nn.Module
TrainData = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, phyre.
                  ActionSimulator, torch.Tensor]


def build_model(network_type: str, **kwargs) -> NeuralModel:
    """Builds a DQN network by name."""
    if network_type == 'resnet18':
        model = nets.ResNet18FilmAction(
            kwargs['action_space_dim'],
            fusion_place=kwargs['fusion_place'],
            action_hidden_size=kwargs['action_hidden_size'],
            action_layers=kwargs['action_layers'])
    elif network_type == 'simple':
        model = nets.SimpleNetWithAction(kwargs['action_space_dim'])
    else:
        raise ValueError('Unknown network type: %s' % network_type)
    return model


def get_latest_checkpoint(output_dir: str) -> Optional[str]:
    known_checkpoints = sorted(glob.glob(os.path.join(output_dir, 'ckpt.*')))
    if known_checkpoints:
        return known_checkpoints[-1]
    else:
        return None


def load_agent_from_folder(agent_folder: str) -> NeuralModel:
    last_checkpoint = get_latest_checkpoint(agent_folder)
    assert last_checkpoint is not None, agent_folder
    # logging.info('Loading a model from: %s', last_checkpoint)
    print(f'Loading a model from: {last_checkpoint}')
    last_checkpoint = torch.load(last_checkpoint)
    model = build_model(**last_checkpoint['model_kwargs'])
    model.load_state_dict(last_checkpoint['model'])
    model.to('cuda')
    return model


def train(output_dir, task_ids, cache, train_batch_size, learning_rate, updates, negative_sampling_prob,
          save_checkpoints_every, fusion_place, network_type, balance_classes, num_auccess_actions, 
          eval_every, action_layers, action_hidden_size, cosine_scheduler, dev_tasks_ids=None):

    train_set = PHYRECls(task_ids=task_ids, dev_task_ids=dev_tasks_ids)

    assert not balance_classes or (negative_sampling_prob == 1), (
        balance_classes, negative_sampling_prob)

    device = 'cuda'
    model_kwargs = dict(network_type=network_type,
                        action_space_dim=train_set.simulator.action_space_dim,
                        fusion_place=fusion_place,
                        action_hidden_size=action_hidden_size,
                        action_layers=action_layers)
    model = build_model(**model_kwargs)
    model.train()
    model.to(device)
    # logging.info(model)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if cosine_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates)
    else:
        scheduler = None
    # logging.info('Starting actual training for %d updates', updates)
    print(f'Starting actual training for {updates} updates')

    rng = np.random.RandomState(42)

    last_checkpoint = get_latest_checkpoint(output_dir)
    if last_checkpoint is not None:
        logging.info('Going to load from %s', last_checkpoint)
        last_checkpoint = torch.load(last_checkpoint)
        model.load_state_dict(last_checkpoint['model'])
        optimizer.load_state_dict(last_checkpoint['optim'])
        rng.set_state(last_checkpoint['rng'])
        if scheduler is not None:
            scheduler.load_state_dict(last_checkpoint['scheduler'])

    def print_eval_stats(batch_id):
        # logging.info('Start eval')
        print('Start eval')
        eval_batch_size = train_batch_size * 4
        stats = {}
        stats['batch_id'] = batch_id + 1
        stats['train_loss'] = eval_loss(model, train_set.eval_train, eval_batch_size)
        if train_set.eval_dev:
            stats['dev_loss'] = eval_loss(model, train_set.eval_dev, eval_batch_size)
        if num_auccess_actions > 0:
            # logging.info('Start AUCCESS eval')
            print('Start AUCCESS eval')
            # stats['train_auccess'] = _eval_and_score_actions(
            #     cache, model, train_set.eval_train[3], num_auccess_actions,
            #     eval_batch_size, train_set.eval_train[4])
            if train_set.eval_dev:
                stats['dev_auccess'] = _eval_and_score_actions(
                    cache, model, train_set.eval_dev[3], num_auccess_actions,
                    eval_batch_size, train_set.eval_dev[4])

        # logging.info('__log__: %s', stats)
        print(f'__log__: {stats}')

    report_every = 1000
    # logging.info('Report every %d; eval every %d', report_every, eval_every)
    print(f'Report every {report_every}; eval every {eval_every}')
    if save_checkpoints_every > eval_every:
        save_checkpoints_every -= save_checkpoints_every % eval_every

    losses = []
    acc = [0, 0, 0, 0]
    last_time = time.time()
    for batch_id, batch_indices in enumerate(train_set.train_indices_sampler()):
        if batch_id >= updates:
            break

        batch_task_indices, batch_observations, batch_actions, batch_is_solved = train_set.get_data(batch_indices)
        if scheduler is not None:
            scheduler.step()
        model.train()

        batch_observations = batch_observations
        batch_actions = batch_actions.to('cuda')
        batch_is_solved = batch_is_solved.to('cuda')

        optimizer.zero_grad()
        pred = model(batch_observations, batch_actions)
        loss = model.ce_loss(pred, batch_is_solved)

        pred = pred.sigmoid() >= 0.5
        acc[0] += ((pred == batch_is_solved)[batch_is_solved == 1]).sum().item()
        acc[1] += ((pred == batch_is_solved)[batch_is_solved == 0]).sum().item()
        acc[2] += (batch_is_solved == 1).sum().item()
        acc[3] += (batch_is_solved == 0).sum().item()

        loss.backward()
        optimizer.step()
        losses.append(loss.mean().item())
        if save_checkpoints_every > 0:
            if (batch_id + 1) % save_checkpoints_every == 0:
                fpath = os.path.join(output_dir, 'ckpt.%08d' % (batch_id + 1))
                # logging.info('Saving: %s', fpath)
                print(f'Saving: {fpath}')
                torch.save(
                    dict(
                        model_kwargs=model_kwargs,
                        model=model.state_dict(),
                        optim=optimizer.state_dict(),
                        done_batches=batch_id + 1,
                        rng=rng.get_state(),
                        scheduler=scheduler and scheduler.state_dict(),
                    ), fpath)
        if (batch_id + 1) % eval_every == 0:
            print_eval_stats(batch_id)
        if (batch_id + 1) % report_every == 0:
            speed = report_every / (time.time() - last_time)
            last_time = time.time()
            print(f'Iter: {batch_id + 1}, examples: {(batch_id + 1) * train_batch_size}, '
                  f'mean loss: {np.mean(losses[-report_every:]):.3f}, speed: {speed:.1f}'
                  f' lr: {get_lr(optimizer):.5f} p acc: {acc[0] / acc[2]:.4f} n acc: {acc[1] / acc[3]:.4f}')
            acc = [0, 0, 0, 0]
            # logging.debug(
            #     'Iter: %s, examples: %d, mean loss: %f, speed: %.1f batch/sec,'
            #     ' lr: %f', batch_id + 1, (batch_id + 1) * train_batch_size,
            #     np.mean(losses[-report_every:]), speed, get_lr(optimizer))
    return model.cpu()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def eval_loss(model, data, batch_size):
    task_indices, is_solved, actions, _, observations = data
    losses = []
    observations = observations.to(model.device)
    with torch.no_grad():
        model.eval()
        for i in range(0, len(task_indices), batch_size):
            batch_indices = task_indices[i:i + batch_size]
            batch_task_indices = task_indices[batch_indices]
            batch_observations = observations[batch_task_indices]
            batch_actions = actions[batch_indices]
            batch_is_solved = is_solved[batch_indices]
            loss = model.ce_loss(model(batch_observations, batch_actions),
                                 batch_is_solved)
            losses.append(loss.item() * len(batch_indices))
    return sum(losses) / len(task_indices)


def eval_actions(model, actions, batch_size, observations):
    scores = []
    with torch.no_grad():
        model.eval()
        preprocessed = model.preprocess(
            torch.LongTensor(observations).unsqueeze(0))
        for batch_start in range(0, len(actions), batch_size):
            batch_end = min(len(actions), batch_start + batch_size)
            batch_actions = torch.FloatTensor(actions[batch_start:batch_end])
            batch_scores = model(None, batch_actions, preprocessed=preprocessed)
            assert len(batch_scores) == len(batch_actions), (
                batch_actions.shape, batch_scores.shape)
            scores.append(batch_scores.cpu().numpy())
    return np.concatenate(scores)


def _eval_and_score_actions(cache, model, simulator, num_actions, batch_size,
                            observations):
    actions = cache.action_array[:num_actions]
    indices = np.random.RandomState(1).permutation(
        len(observations))[:AUCCESS_EVAL_TASKS]
    evaluator = phyre.Evaluator(
        [simulator.task_ids[index] for index in indices])
    for i, task_index in enumerate(indices):
        tprint(f'{i}/{len(indices)} {task_index}')
        scores = eval_actions(model, actions, batch_size,
                              observations[task_index]).tolist()

        _, sorted_actions = zip(
            *sorted(zip(scores, actions), key=lambda x: (-x[0], tuple(x[1]))))
        for action in sorted_actions:
            if (evaluator.get_attempts_for_task(i) >= phyre.MAX_TEST_ATTEMPTS):
                break
            status = simulator.simulate_action(task_index,
                                               action,
                                               need_images=False).status
            evaluator.maybe_log_attempt(i, status)
    return evaluator.get_aucess()


def arg_parse():
    # only the most general argument is passed here
    # task-specific parameters should be passed by config file
    parser = argparse.ArgumentParser(description='RPIN parameters')
    parser.add_argument('--output', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    output_dir = f'outputs/phys/PHYRECls/within/{args.output}'
    os.makedirs(output_dir, exist_ok=True)

    action_tier_name = 'ball'

    eval_setup = 'ball_within_template'
    train_tasks, dev_tasks, test_ids = phyre.get_fold(eval_setup, 1)
    task_ids = train_tasks + dev_tasks
    dev_tasks_ids = test_ids

    cache = phyre.get_default_100k_cache(action_tier_name)

    train_batch_size = 64
    learning_rate = 0.0003
    max_train_actions = None
    updates = 100000
    negative_sampling_prob = 1.0
    save_checkpoints_every = 10000
    fusion_place = 'last'
    network_type = 'resnet18'
    balance_classes = 1
    num_auccess_actions = 10000
    eval_every = 20000
    action_layers = 1
    action_hidden_size = 256
    cosine_scheduler = 1
    train(output_dir, task_ids, cache, train_batch_size, learning_rate, updates, negative_sampling_prob,
          save_checkpoints_every, fusion_place, network_type, balance_classes, num_auccess_actions,
          eval_every, action_layers, action_hidden_size, cosine_scheduler, dev_tasks_ids)
