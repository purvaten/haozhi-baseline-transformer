import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.roi_align import RoIAlign

from neuralphys.utils.config import _C as C
from neuralphys.models.layers.CIN import InterNet
from neuralphys.models.backbones.build import build_backbone


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # a bunch of temporary flag, the useful setting will be merge to config file
        # here is just to easily setup experiments
        self.use_ln = False
        self.norm_before_relu = False
        self.pos_feat_ar = False
        # define private variables
        self.time_step = C.RIN.INPUT_SIZE
        self.ve_feat_dim = C.RIN.VE_FEAT_DIM  # visual encoder feature dimension
        self.in_feat_dim = C.RIN.IN_FEAT_DIM  # interaction net feature dimension
        self.num_objs = C.RIN.NUM_OBJS
        self.mask_size = C.RIN.MASK_SIZE
        self.po_feat_dim = self.in_feat_dim if C.RIN.COOR_FEATURE else 0  # position feature dimension

        # build image encoder
        self.backbone = build_backbone(C.RIN.BACKBONE, self.ve_feat_dim, C.INPUT.IMAGE_CHANNEL)

        # extract object feature -> convert to object state
        pool_size = C.RIN.ROI_POOL_SIZE
        self.roi_align = RoIAlign(
            (pool_size, pool_size),
            spatial_scale=C.RIN.ROI_POOL_SPATIAL_SCALE,
            sampling_ratio=C.RIN.ROI_POOL_SAMPLE_R,
        )
        roi2state = [nn.Conv2d(self.ve_feat_dim, self.in_feat_dim, kernel_size=3, padding=1),
                     nn.ReLU(inplace=True)]
        assert C.RIN.N_EXTRA_ROI_F > 0
        for _ in range(C.RIN.N_EXTRA_ROI_F):
            roi2state.append(nn.Conv2d(self.ve_feat_dim, self.in_feat_dim,
                                       kernel_size=C.RIN.EXTRA_F_KERNEL, stride=1, padding=C.RIN.EXTRA_F_PADDING))
            if self.norm_before_relu and _ == C.RIN.N_EXTRA_ROI_F - 1:
                continue
            roi2state.append(nn.ReLU(inplace=True))
        self.roi2state = nn.Sequential(*roi2state)

        graph = []
        for i in range(self.time_step):
            graph.append(InterNet(self.in_feat_dim))
        self.graph = nn.ModuleList(graph)

        assert C.RIN.N_EXTRA_PRED_F == 0
        if self.norm_before_relu:
            predictor = [nn.Conv2d(self.in_feat_dim * self.time_step, self.in_feat_dim, kernel_size=1)]
        else:
            predictor = [nn.Conv2d(self.in_feat_dim * self.time_step, self.in_feat_dim, kernel_size=1), nn.ReLU()]

        for _ in range(C.RIN.N_EXTRA_PRED_F):
            predictor.append(nn.Conv2d(self.in_feat_dim, self.in_feat_dim,
                                       kernel_size=C.RIN.EXTRA_F_KERNEL, stride=1, padding=C.RIN.EXTRA_F_PADDING))
            predictor.append(nn.ReLU(inplace=True))
        self.predictor = nn.Sequential(*predictor)

        self.decoder_output = 4
        self.bbox_decoder = nn.Linear(self.in_feat_dim * pool_size * pool_size, self.decoder_output)

        if C.RIN.COOR_FEATURE:
            self.pos_encoder = nn.Sequential(
                nn.Linear(4, self.po_feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.po_feat_dim, self.po_feat_dim),
                nn.ReLU(inplace=True),
            )
            if self.norm_before_relu:
                self.pos_merger = nn.Sequential(
                    nn.Conv2d(self.in_feat_dim + self.po_feat_dim, self.in_feat_dim, kernel_size=3, stride=1,
                              padding=1),
                )
            else:
                self.pos_merger = nn.Sequential(
                    nn.Conv2d(self.in_feat_dim + self.po_feat_dim, self.in_feat_dim, kernel_size=3, stride=1,
                              padding=1),
                    nn.ReLU(inplace=True),
                )
        
        if C.RIN.MASK_LOSS_WEIGHT > 0:
            self.mask_decoder = nn.Sequential(
                nn.Linear(self.in_feat_dim * pool_size * pool_size, self.in_feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.in_feat_dim, self.mask_size * self.mask_size),
                nn.Sigmoid(),
            )

        if C.RIN.SEQ_CLS_LOSS_WEIGHT > 0:
            self.seq_feature = nn.Sequential(
                nn.Linear(self.in_feat_dim * pool_size * pool_size, self.in_feat_dim * 4),
                nn.ReLU(inplace=True),
                nn.Linear(self.in_feat_dim * 4, self.in_feat_dim),
                nn.ReLU(inplace=True),
            )
            self.seq_score = nn.Sequential(
                nn.Linear(self.in_feat_dim, 1),
                nn.Sigmoid()
            )

        if self.use_ln:
            norms = [nn.LayerNorm([self.in_feat_dim, 5, 5]) for _ in range(self.time_step)]
            self.norms = nn.ModuleList(norms)

    def forward(self, x, rois, p_feat, num_rollouts=10, g_idx=None, x_t=None, phase='train'):
        self.num_objs = rois.shape[2]
        # x: (b, t, c, h, w)
        # reshape time to batch dimension
        batch_size, time_step = x.shape[:2]
        assert self.time_step == time_step
        # of shape (b, t, o, dim)
        x = self.extract_object_feature(x, rois)

        if C.RIN.COOR_FEATURE:
            x = self.attach_position_embedding(x, p_feat)
        
        bbox_rollout = []
        mask_rollout = []
        state_list = [x[:, i] for i in range(self.time_step)]
        for i in range(num_rollouts):
            if self.use_ln:
                state_list = [self.norms[j](state_list[j]) for j in range(self.time_step)]

            if self.norm_before_relu:
                c = [self.graph[j](F.relu(state_list[j]), g_idx) for j in range(self.time_step)]
            else:
                c = [self.graph[j](state_list[j], g_idx) for j in range(self.time_step)]

            all_c = torch.cat(c, 2)
            s = self.predictor(all_c.reshape((-1,) + (all_c.shape[-3:])))
            s = s.reshape((batch_size, self.num_objs) + s.shape[-3:])

            if self.norm_before_relu:
                bbox = self.bbox_decoder(F.relu(s.reshape(batch_size, self.num_objs, -1)))
                if C.RIN.MASK_LOSS_WEIGHT:
                    mask = self.mask_decoder(F.relu(s.reshape(batch_size, self.num_objs, -1)))
                    mask_rollout.append(mask)
            else:
                bbox = self.bbox_decoder(s.reshape(batch_size, self.num_objs, -1))
                if C.RIN.MASK_LOSS_WEIGHT:
                    mask = self.mask_decoder(s.reshape(batch_size, self.num_objs, -1))
                    mask_rollout.append(mask)

            bbox_rollout.append(bbox)

            if C.RIN.COOR_FEATURE and self.pos_feat_ar:
                # un-squeeze time dimension
                s = self.attach_position_embedding(s[:, None], bbox[:, None])
                s = s.squeeze(1)

            state_list = state_list[1:] + [s]

        # final timestep feature (b, o, dim, psz, psz)
        # transform to a fix dim (b, o, dim * 4)
        # average (b, dim * 4)
        # classifier
        seq_score = []
        if C.RIN.SEQ_CLS_LOSS_WEIGHT > 0:
            seq_feature = self.seq_feature(state_list[-1].reshape(batch_size, self.num_objs, -1))
            valid_seq = g_idx[:, ::C.RIN.NUM_OBJS - 1, [2]]
            seq_feature = (seq_feature * valid_seq).sum(dim=1) / valid_seq.sum(dim=1)
            seq_score = self.seq_score(seq_feature)

        bbox_rollout = torch.stack(bbox_rollout).permute(1, 0, 2, 3)
        bbox_rollout = bbox_rollout.reshape(-1, num_rollouts, self.num_objs, self.decoder_output)

        if len(mask_rollout) > 0:
            mask_rollout = torch.stack(mask_rollout).permute(1, 0, 2, 3)
            mask_rollout = mask_rollout.reshape(-1, num_rollouts, self.num_objs, self.mask_size, self.mask_size)

        outputs = {
            'boxes': bbox_rollout,
            'masks': mask_rollout,
            'score': seq_score,
        }
        return outputs

    def attach_position_embedding(self, x, p_feat):
        # x is of shape (B, t, o, dim, psz, psz)
        p_feat = self.pos_encoder(p_feat)
        p_feat = p_feat[..., None, None].repeat(1, 1, 1, 1, C.RIN.ROI_POOL_SIZE, C.RIN.ROI_POOL_SIZE)
        x = torch.cat([x, p_feat], dim=3)
        b, t, o, d, sz, sz = x.shape
        x = self.pos_merger(x.reshape(b * t * o, d, sz, sz))
        x = x.reshape(b, t, o, -1, sz, sz)
        return x

    def extract_object_feature(self, x, rois):
        # visual feature, comes from RoI Pooling
        # RPIN method:
        batch_size, time_step = x.shape[0], x.shape[1]
        x = x.reshape((batch_size * time_step,) + x.shape[2:])  # (b x t, c, h, w)
        x = self.backbone(x)
        roi_pool = self.roi_align(x, rois.reshape(-1, 5))  # (b * t * num_objs, feat_dim)
        roi_pool = self.roi2state(roi_pool)
        x = roi_pool.reshape((batch_size, time_step, self.num_objs) + (roi_pool.shape[-3:]))
        return x
