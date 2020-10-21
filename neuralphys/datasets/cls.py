import phyre
import torch
import numpy as np
from typing import Sequence, Tuple

from neuralphys.utils.misc import tprint

AUCCESS_EVAL_TASKS = 1e6
XE_EVAL_SIZE = 10000
max_a = 100000

TaskIds = Sequence[str]
NeuralModel = torch.nn.Module
TrainData = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, phyre.
                  ActionSimulator, torch.Tensor]


def create_balanced_eval_set(cache: phyre.SimulationCache, task_ids: TaskIds,
                             size: int, tier: str) -> TrainData:
    """Prepares balanced eval set to run through a network.

    Selects (size // 2) positive (task, action) pairs and (size // 2)
    negative pairs and represents them in a compact formaer

    Returns a tuple
        (task_indices, is_solved, selected_actions, simulator, observations).

        Tensors task_indices, is_solved, selected_actions, observations, all
        have lengths size and correspond to some (task, action) pair.
        For any i the following is true:
            is_solved[i] is true iff selected_actions[i] solves task
            task_ids[task_indices[i]].
    """
    task_ids = tuple(task_ids)
    data = cache.get_sample(task_ids)
    actions = data['actions'][:max_a]
    simulation_statuses = data['simulation_statuses'][:, :max_a]

    flat_statuses = simulation_statuses.reshape(-1)
    [positive_indices] = (flat_statuses == int(phyre.SimulationStatus.SOLVED)).nonzero()
    [negative_indices] = (flat_statuses == int(phyre.SimulationStatus.NOT_SOLVED)).nonzero()

    half_size = size // 2
    rng = np.random.RandomState(42)
    positive_indices = rng.choice(positive_indices, half_size)
    negative_indices = rng.choice(negative_indices, half_size)

    all_indices = np.concatenate([positive_indices, negative_indices])
    selected_actions = torch.FloatTensor(actions[all_indices % len(actions)])
    is_solved = torch.LongTensor(flat_statuses[all_indices].astype('int')) > 0
    task_indices = torch.LongTensor(all_indices // len(actions))

    simulator = phyre.initialize_simulator(task_ids, tier)
    observations = torch.LongTensor(simulator.initial_scenes)
    return task_indices, is_solved, selected_actions, simulator, observations


def compact_simulation_data_to_trainset(action_tier_name: str,
                                        actions: np.ndarray,
                                        simulation_statuses: Sequence[int],
                                        task_ids: TaskIds) -> TrainData:
    """Converts result of SimulationCache.get_data() to pytorch tensors.

    The format of the output is the same as in create_balanced_eval_set.
    """
    invalid = int(phyre.SimulationStatus.INVALID_INPUT)
    solved = int(phyre.SimulationStatus.SOLVED)
    task_indices = np.repeat(np.arange(len(task_ids)).reshape((-1, 1)),
                             actions.shape[0],
                             axis=1).reshape(-1)
    action_indices = np.repeat(np.arange(actions.shape[0]).reshape((1, -1)),
                               len(task_ids),
                               axis=0).reshape(-1)
    simulation_statuses = simulation_statuses.reshape(-1)

    good_statuses = simulation_statuses != invalid
    is_solved = (torch.LongTensor(simulation_statuses[good_statuses].astype('uint8')) == solved).numpy()
    action_indices = action_indices[good_statuses]
    actions = actions[action_indices]
    task_indices = task_indices[good_statuses]

    # whether to enforce maximum actions:
    # could comment below
    max_pos = 50
    max_neg = 200
    new_is_solved = []
    new_actions = []
    new_task_indices = []
    for i in range(2000):
        tprint(i)
        start_id = np.where(task_indices == i)[0][0]
        end_id = np.where(task_indices == i)[0][-1] + 1

        pos_indices = np.where(is_solved[start_id:end_id] == 1)[0]
        np.random.shuffle(pos_indices)
        pos_indices = pos_indices[:max_pos]

        neg_indices = np.where(is_solved[start_id:end_id] == 0)[0]
        np.random.shuffle(neg_indices)
        neg_indices = neg_indices[:max_neg]

        indices = np.concatenate((pos_indices, neg_indices))
        new_is_solved.append(is_solved[start_id:end_id][indices])
        new_actions.append(actions[start_id:end_id][indices])
        new_task_indices.append(task_indices[start_id:end_id][indices])

    actions = np.concatenate(new_actions)
    task_indices = np.concatenate(new_task_indices)
    is_solved = np.concatenate(new_is_solved)
    # could comment above

    actions = torch.FloatTensor(actions)
    task_indices = torch.LongTensor(task_indices)
    is_solved = torch.BoolTensor(is_solved)
    simulator = phyre.initialize_simulator(task_ids, action_tier_name)
    observations = torch.LongTensor(simulator.initial_scenes)

    return task_indices, is_solved, actions, simulator, observations


class PHYRECls:
    def __init__(self, task_ids, dev_task_ids, tier='ball', image_ext='.jpg'):
        self.image_ext = image_ext

        cache = phyre.get_default_100k_cache(tier)
        training_data = cache.get_sample(task_ids, None)

        print('gg')
        training_data['actions'] = training_data['actions'][:max_a]
        training_data['simulation_statuses'] = training_data['simulation_statuses'][:, :max_a]
        print('Preparing training data')
        task_indices, is_solved, actions, simulator, observations = (
            compact_simulation_data_to_trainset(tier, **training_data))

        self.task_indices = task_indices
        self.is_solved = is_solved.pin_memory()
        self.actions = actions.pin_memory()
        self.observations = observations.to('cuda')
        self.simulator = simulator
        print('Creating eval subset from train')
        self.eval_train = create_balanced_eval_set(cache, simulator.task_ids, XE_EVAL_SIZE, tier)
        if dev_task_ids is not None:
            # logging.info('Creating eval subset from dev')
            print('Creating eval subset from dev')
            self.eval_dev = create_balanced_eval_set(cache, dev_task_ids, XE_EVAL_SIZE, tier)
        else:
            self.eval_dev = None

        self.rng = np.random.RandomState(42)
        self.train_batch_size = 64

    def train_indices_sampler(self):
        indices = np.arange(len(self.is_solved))
        solved_mask = self.is_solved.numpy() > 0
        positive_indices = indices[solved_mask]
        negative_indices = indices[~solved_mask]
        positive_size = self.train_batch_size // 2
        while True:
            positives = self.rng.choice(positive_indices, size=positive_size)
            negatives = self.rng.choice(negative_indices, size=self.train_batch_size - positive_size)
            positive_size = self.train_batch_size - positive_size
            yield np.concatenate((positives, negatives))

    def get_data(self, batch_indices):
        batch_task_indices = self.task_indices[batch_indices]
        batch_observations = self.observations[batch_task_indices]
        batch_actions = self.actions[batch_indices]
        batch_is_solved = self.is_solved[batch_indices]
        return batch_task_indices, batch_observations, batch_actions, batch_is_solved
