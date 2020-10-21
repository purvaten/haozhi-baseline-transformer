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

import torch
import torch.nn as nn
import torchvision

import phyre


class ResNet18Film(nn.Module):

    def __init__(self, data_format, fuse):
        super().__init__()
        self.data_format = data_format
        self.fuse = fuse
        net = torchvision.models.resnet18(pretrained=False)

        # temporary variable
        self.num_colors = 7
        self.with_dropout = False

        if self.fuse == 'early':
            conv1 = nn.Conv2d(self.num_colors * data_format, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.reason = nn.Linear(512, 1)
        elif self.fuse == 'late':
            conv1 = nn.Conv2d(self.num_colors, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.reason = nn.Linear(512 * data_format, 1)
        elif self.fuse == '3d':
            conv1 = nn.Conv2d(self.num_colors, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.fuse_net = nn.Sequential(
                nn.Conv3d(512, 512, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=True),
                # nn.ReLU(True),
                # nn.Conv3d(512, 512, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=True),
                # nn.ReLU(True),
                # nn.Conv3d(512, 512, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=True),
                # nn.ReLU(True),
            )
            self.reason = nn.Linear(512, 1)
        else:
            raise NotImplementedError

        self.register_buffer('embed_weights', torch.eye(self.num_colors))
        self.stem = nn.Sequential(conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList([
            net.layer1, net.layer2, net.layer3, net.layer4
        ])

        if self.with_dropout:
            self.dropout = nn.Dropout(p=0.1, inplace=True)

    @property
    def device(self):
        if hasattr(self, 'parameters') and next(self.parameters()).is_cuda:
            return 'cuda'
        else:
            return 'cpu'

    def forward(self, observations):
        assert observations.ndim == 4
        batch_size = observations.shape[0]
        time_step = observations.shape[1]

        image = observations
        
        if self.num_colors == 7:
            image = image.reshape((-1,) + observations.shape[2:])
            image = self._image_colors_to_onehot(image)

        if self.fuse == 'early':
            image = image.reshape(batch_size, self.num_colors * self.data_format, image.shape[2], image.shape[3])

        features = self.stem(image)
        for stage in self.stages:
            features = stage(features)
        
        if self.fuse == '3d':
            # n x t x c x h x w
            features = features.reshape(batch_size, time_step, 512, 4, 4).transpose(1, 2)
            features = self.fuse_net(features)
            features = nn.functional.adaptive_max_pool3d(features, 1)
        else:
            features = nn.functional.adaptive_max_pool2d(features, 1)
        features = features.flatten(1)
        features = features.reshape(batch_size, -1)
        if self.with_dropout:
            features = self.dropout(features)
        return self.reason(features).squeeze(-1)

    def ce_loss(self, decisions, targets):
        targets = targets.to(dtype=torch.float, device=decisions.device)
        return nn.functional.binary_cross_entropy_with_logits(
            decisions, targets)

    def _image_colors_to_onehot(self, indices):
        onehot = torch.nn.functional.embedding(
            indices.to(dtype=torch.long, device=self.embed_weights.device),
            self.embed_weights)
        onehot = onehot.permute(0, 3, 1, 2).contiguous()
        return onehot


class ResNet18FilmAction(nn.Module):

    def __init__(self,
                 action_size,
                 action_layers=1,
                 action_hidden_size=256,
                 fusion_place='last'):
        super().__init__()
        net = torchvision.models.resnet18(pretrained=False)
        conv1 = nn.Conv2d(phyre.NUM_COLORS,
                          64,
                          kernel_size=7,
                          stride=2,
                          padding=3,
                          bias=False)
        self.register_buffer('embed_weights', torch.eye(phyre.NUM_COLORS))
        self.stem = nn.Sequential(conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList(
            [net.layer1, net.layer2, net.layer3, net.layer4])

        def build_film(output_size):
            return FilmActionNetwork(action_size,
                                     output_size,
                                     hidden_size=action_hidden_size,
                                     num_layers=action_layers)

        assert fusion_place in ('first', 'last', 'all', 'none', 'last_single')

        self.last_network = None
        if fusion_place == 'all':
            self.action_networks = nn.ModuleList(
                [build_film(size) for size in (64, 64, 128, 256)])
        elif fusion_place == 'last':
            # Save module as attribute.
            self._action_network = build_film(256)
            self.action_networks = [None, None, None, self._action_network]
        elif fusion_place == 'first':
            # Save module as attribute.
            self._action_network = build_film(64)
            self.action_networks = [self._action_network, None, None, None]
        elif fusion_place == 'last_single':
            # Save module as attribute.
            self.last_network = build_film(512)
            self.action_networks = [None, None, None, None]
        elif fusion_place == 'none':
            self.action_networks = [None, None, None, None]
        else:
            raise Exception('Unknown fusion place: %s' % fusion_place)
        self.reason = nn.Linear(512, 1)

    @property
    def device(self):
        if hasattr(self, 'parameters') and next(self.parameters()).is_cuda:
            return 'cuda'
        else:
            return 'cpu'

    def preprocess(self, observations):
        image = self._image_colors_to_onehot(observations)
        features = self.stem(image)
        for stage, act_layer in zip(self.stages, self.action_networks):
            if act_layer is not None:
                break
            features = stage(features)
        else:
            features = nn.functional.adaptive_max_pool2d(features, 1)
        return dict(features=features)

    def forward(self, observations, actions, preprocessed=None):
        if preprocessed is None:
            preprocessed = self.preprocess(observations)
        return self._forward(actions, **preprocessed)

    def _forward(self, actions, features):
        actions = actions.to(features.device)
        skip_compute = True
        for stage, film_layer in zip(self.stages, self.action_networks):
            if film_layer is not None:
                skip_compute = False
                features = film_layer(actions, features)
            if skip_compute:
                continue
            features = stage(features)
        if not skip_compute:
            features = nn.functional.adaptive_max_pool2d(features, 1)
        if self.last_network is not None:
            features = self.last_network(actions, features)
        features = features.flatten(1)
        if features.shape[0] == 1 and actions.shape[0] != 1:
            # Haven't had a chance to use actions. So will match batch size as
            # in actions manually.
            features = features.expand(actions.shape[0], -1)
        return self.reason(features).squeeze(-1)

    def ce_loss(self, decisions, targets):
        targets = targets.to(dtype=torch.float, device=decisions.device)
        return nn.functional.binary_cross_entropy_with_logits(
            decisions, targets)

    def _image_colors_to_onehot(self, indices):
        onehot = torch.nn.functional.embedding(
            indices.to(dtype=torch.long, device=self.embed_weights.device),
            self.embed_weights)
        onehot = onehot.permute(0, 3, 1, 2).contiguous()
        return onehot


def _image_colors_to_onehot(indices):
    onehot = torch.nn.functional.embedding(
        indices, torch.eye(phyre.NUM_COLORS, device=indices.device))
    onehot = onehot.pertmute(0, 3, 1, 2).contiguous()
    return onehot
