import torch
from torch import nn
import torch.nn.functional as F

from neuralphys.utils.config import _C as C
import pdb

# promised to be temporary flags
trans = 1
conv = 1


class InterNet(nn.Module):
    def __init__(self, in_feat_dim):
        super(InterNet, self).__init__()
        self.in_feat_dim = in_feat_dim
        # self dynamics, input object state, output new object state
        self_dynamics = [
            nn.Conv2d(self.in_feat_dim, self.in_feat_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)
        ]
        for _ in range(C.RIN.N_EXTRA_SELFD_F):
            self_dynamics.append(nn.Conv2d(self.in_feat_dim, self.in_feat_dim,
                                           kernel_size=C.RIN.EXTRA_F_KERNEL, stride=1, padding=C.RIN.EXTRA_F_PADDING))
            self_dynamics.append(nn.ReLU(inplace=True))
        self.self_dynamics = nn.Sequential(*self_dynamics)
        # relation dynamics, input pairwise object states, output new object state
        rel_dynamics = [
            nn.Conv2d(self.in_feat_dim * 2, self.in_feat_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)
        ]
        for _ in range(C.RIN.N_EXTRA_RELD_F):
            rel_dynamics.append(nn.Conv2d(self.in_feat_dim, self.in_feat_dim,
                                          kernel_size=C.RIN.EXTRA_F_KERNEL, stride=1, padding=C.RIN.EXTRA_F_PADDING))
            rel_dynamics.append(nn.ReLU(inplace=True))
        self.rel_dynamics = nn.Sequential(*rel_dynamics)
        # affector
        affector = [
            nn.Conv2d(self.in_feat_dim, self.in_feat_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)
        ]
        for _ in range(C.RIN.N_EXTRA_AFFECTOR_F):
            affector.append(nn.Conv2d(self.in_feat_dim, self.in_feat_dim,
                                      kernel_size=C.RIN.EXTRA_F_KERNEL, stride=1, padding=C.RIN.EXTRA_F_PADDING))
            affector.append(nn.ReLU(inplace=True))
        self.affector = nn.Sequential(*affector)
        # aggregator
        aggregator = [
            nn.Conv2d(self.in_feat_dim * 2, self.in_feat_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)
        ]
        for _ in range(C.RIN.N_EXTRA_AGGREGATOR_F):
            aggregator.append(nn.Conv2d(self.in_feat_dim, self.in_feat_dim,
                                        kernel_size=C.RIN.EXTRA_F_KERNEL, stride=1, padding=C.RIN.EXTRA_F_PADDING))
            aggregator.append(nn.ReLU(inplace=True))
        self.aggregator = nn.Sequential(*aggregator)
        # self-attention
        self.downsize = 32
        self.temperature = (self.downsize * 5 * 5) ** 0.5
        self.attn_conv2d = nn.Conv2d(self.in_feat_dim, self.downsize, 1)
        self.w_qs = nn.Linear(self.downsize * 5 * 5, self.downsize * 5 * 5, bias=False)
        self.w_ks = nn.Linear(self.downsize * 5 * 5, self.downsize * 5 * 5, bias=False)
        # transformer
        self.layer_norm = nn.LayerNorm([C.RIN.NUM_OBJS, self.in_feat_dim, 5, 5], eps=1e-6)
        # conv
        self.convq = nn.Sequential(*[nn.Conv2d(self.downsize, self.downsize, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)])
        self.convk = nn.Sequential(*[nn.Conv2d(self.downsize, self.downsize, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)])

    def forward(self, x, g_idx=None):
        s = x    # (64, 6, 256, 5, 5)
        # of shape (b, o, dim, 7, 7)
        batch_size, num_objs, dim, psz, psz = x.shape
        x1 = x.repeat(1, num_objs - 1, 1, 1, 1)    # (64, 30, 256, 5, 5)
        i1 = g_idx[..., [0], None, None].repeat(1, 1, dim, psz, psz)    # (64, 30, 256, 5, 5)
        y1 = torch.gather(x1, 1, i1)    # (64, 30, 256, 5, 5)
        i2 = g_idx[..., [1], None, None].repeat(1, 1, dim, psz, psz)    # (64, 30, 256, 5, 5)
        y2 = torch.gather(x1, 1, i2)    # (64, 30, 256, 5, 5)
        r = torch.cat([y1, y2], dim=2)    # (64, 30, 512, 5, 5)
        r = r * g_idx[:, :, [2], None, None]    # (64, 30, 512, 5, 5)
        r = r.reshape(-1, dim * 2, psz, psz)    # (1920, 512, 5, 5)
        r = self.rel_dynamics(r)    # (1920, 512, 5, 5)
        r = r.reshape(batch_size, num_objs, num_objs - 1, dim, psz, psz)    # (64, 6, 5, 256, 5, 5)

        # self-attention weights - rel_wts: (64, 6, 5), self_wts: (64, 6)
        s2d = self.attn_conv2d(s.reshape(-1, dim, psz, psz))    # (64 * 6, 32, 5, 5)
        if conv:
            q = self.convq(s2d).reshape(-1, num_objs, self.downsize * psz * psz)    # (64, 6, 800)
            k = self.convk(s2d).reshape(-1, num_objs, self.downsize * psz * psz)    # (64, 6, 800)
        else:
            s2d_flat = s2d.reshape(-1, self.downsize * psz * psz)    # (64 * 6, 800)
            q = self.w_qs(s2d_flat).reshape(-1, num_objs, self.downsize * psz * psz)    # (64, 6, 800)
            k = self.w_ks(s2d_flat).reshape(-1, num_objs, self.downsize * psz * psz)    # (64, 6, 800)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 1))    # (64, 6, 6)
        softmax = F.softmax(attn, dim=-1)    # (64, 6, 6)
        self_wts = torch.diagonal(softmax, offset=0, dim1=1, dim2=2)    # (64, 6)
        indices = torch.LongTensor([[i for i in range(softmax.shape[1])] for _ in range(softmax.shape[0])]).unsqueeze(1).to('cuda')
        rel_wts = softmax[torch.ones_like(softmax).scatter_(1, indices, 0.).bool()].view(-1, num_objs, num_objs-1)

        r = rel_wts[:, :, :, None, None, None] * r
        r = r.sum(dim=2)    # (64, 6, 256, 5, 5)        

        x = self.self_dynamics(x.reshape(-1, dim, psz, psz)).reshape(batch_size, num_objs, dim, psz, psz)    # (64, 6, 256, 5, 5)
        x = self_wts[:, :, None, None, None] * x

        pred = x + r    # (64, 6, 256, 5, 5)
        a = self.affector(pred.reshape(-1, dim, psz, psz)).reshape(batch_size, num_objs, dim, psz, psz)    # fA in eqn (1) -- (64, 6, 256, 5, 5)
        
        # transformer: add residual to a
        if trans:
            a = a + s
            a = self.layer_norm(a)

        a = torch.cat([a, s], 2)    # (64, 6, 512, 5, 5)
        out = self.aggregator(a.reshape(-1, dim * 2, psz, psz)).reshape(batch_size, num_objs, dim, psz, psz)    # fZ in eqn (1) -- (64, 6, 256, 5, 5)
        return out
