import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_gather
import numpy as np


def cos_angle(v1, v2):
    """
        V1, V2: (N, 3)
        return: (N,)
    """
    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)


class MLPNet_linear(nn.Module):
    def __init__(self,
                 d_in=3,
                 d_mid=256,
                 d_out=1,
                 n_mid=8,
                 bias=0.5,
                 geometric_init=True,
                 weight_norm=False,
                 inside_grad=True,
            ):
        super(MLPNet_linear, self).__init__()
        assert n_mid > 3
        dims = [d_in] + [d_mid for _ in range(n_mid)] + [d_out]
        self.num_layers = len(dims)
        self.skip_in = [n_mid // 2]

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - d_in
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)
            if geometric_init:
                if l == self.num_layers - 2:
                    if inside_grad:  # inside SDF > 0
                        nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        nn.init.constant_(lin.bias, bias)
                    else:
                        nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        nn.init.constant_(lin.bias, -bias)
                else:
                    nn.init.normal_(lin.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(out_dim))
                    nn.init.constant_(lin.bias, 0.0)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

    def forward(self, pos):
        """
            pos: (*, N, C)
        """
        x = pos
        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                x = torch.cat([x, pos], dim=-1)
                x = x / np.sqrt(2)

            lin = getattr(self, "lin" + str(l))
            x = lin(x)

            if l < self.num_layers - 2:
                x = F.relu(x)
        return x

    def gradient(self, x):
        """
            x: (*, N, C), with requires_grad is set to true
        """
        y = self.forward(x)             # (*, N, 1), signed distance

        # y.sum().backward(retain_graph=True)
        # grad_out = x.grad.detach()

        grad_outputs = torch.ones_like(y, requires_grad=False, device=y.device)
        grad_out = torch.autograd.grad(outputs=y,
                                    inputs=x,
                                    grad_outputs=grad_outputs,
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]
        grad_norm = F.normalize(grad_out, dim=-1)     # (*, N, 3)
        return y, grad_norm


class Network(nn.Module):
    def __init__(self, num_points, num_knn):
        super(Network, self).__init__()
        self.num_points = num_points
        self.num_knn = num_knn
        self.num_iter = 2

        self.net = MLPNet_linear(d_in=3, d_mid=256, d_out=1, n_mid=8)

    def forward(self, pcl_source):
        """
            pcl_source: (*, N, 3)
        """
        self.sd_all = []
        self.grad_all = []
        with torch.set_grad_enabled(True):
            pcl_source.requires_grad = True
            sd_temp = torch.zeros_like(pcl_source)[::,0:1]
            grad_temp = torch.zeros_like(pcl_source)

            for i in range(self.num_iter):
                pcl_source = pcl_source - sd_temp * grad_temp

                sd_temp, grad_temp = self.net.gradient(pcl_source)     # (*, N, 1), (*, N, 3)
                self.sd_all.append(sd_temp)
                self.grad_all.append(grad_temp)

                if i == 0:
                    self.sd = sd_temp
                    self.grad_norm = grad_temp
                elif i == 1:
                    self.sd1 = sd_temp
                    self.grad_norm1 = grad_temp
                elif i == 2:
                    self.sd2 = sd_temp
                    self.grad_norm2 = grad_temp
                else:
                    raise ValueError('Not set value')

            self.grad_sum = F.normalize(sum(self.grad_all), dim=-1)

        return self.grad_sum

    def get_loss(self, pcl_raw=None, pcl_source=None, knn_idx=None):
        """
            pcl_raw: (1, M, 3), M >= N
            pcl_source: (1, N+n, 3)
            normal_gt: (1, N, 3)
            knn_idx: (1, N, K)
        """
        num_points = self.num_points
        _device, _dtype = pcl_source.device, pcl_source.dtype
        loss_d = torch.zeros(1, device=_device, dtype=_dtype)
        loss_v1 = torch.zeros(1, device=_device, dtype=_dtype)
        loss_v2 = torch.zeros(1, device=_device, dtype=_dtype)
        loss_v3 = torch.zeros(1, device=_device, dtype=_dtype)
        loss_reg1 = torch.zeros(1, device=_device, dtype=_dtype)
        loss_reg2 = torch.zeros(1, device=_device, dtype=_dtype)
        loss_con = torch.zeros(1, device=_device, dtype=_dtype)
        loss_sd = torch.zeros(1, device=_device, dtype=_dtype)

        pcl_nn = knn_gather(pcl_raw, knn_idx)                   # (1, N, K, 3)
        v = pcl_source[:, :num_points, None, :3] - pcl_nn       # (1, N, K, 3)
        v1 = v[:,:,:8,:].mean(-2)                               # (1, N, 3)
        v2 = v[:,:,:4,:].mean(-2)                               # (1, N, 3)
        v3 = v[:,:,0,:]                                         # (1, N, 3)

        pcl_target = torch.cat((pcl_nn[:,:,0,:], pcl_source[:, num_points:, :]), dim=-2)

        loss_reg1 = 10 * (self.sd[:, num_points:, :]**2).mean()
        loss_reg2 = 10 * (self.sd1**2).mean() #+ 10 * (self.sd2**2).mean()

        weight = torch.exp(-60 * torch.abs(self.sd)).squeeze()      # (N,)

        loss_v1 = torch.linalg.norm((v1 - (self.sd * self.grad_norm)[:, :num_points, :]), ord=2, dim=-1).mean()
        loss_v2 = torch.linalg.norm((v2 - (self.sd * self.grad_norm)[:, :num_points, :]), ord=2, dim=-1).mean()
        loss_v3 = torch.linalg.norm((v3 - (self.sd * self.grad_norm)[:, :num_points, :]), ord=2, dim=-1).mean()

        pcl_source_new = pcl_source - self.sd * self.grad_norm - self.sd1 * self.grad_norm1 #- self.sd2 * self.grad_norm2
        loss_d = 0.3 * torch.linalg.norm((pcl_source_new - pcl_target), ord=2, dim=-1).mean()

        cos_ang = cos_angle(self.grad_norm[0, :, :], self.grad_norm1[0, :, :])  # (N,)
        # cos_ang1 = cos_angle(self.grad_norm[0, :, :], self.grad_norm2[0, :, :])
        loss_con = 0.01 * (weight * (1 - cos_ang)).mean() #+ 0.01 * (weight * (1 - cos_ang1)).mean()

        # loss_sd = 0.01 * torch.clamp(torch.abs(self.sd + self.sd1)[:, :num_points, :] - torch.linalg.norm(v3, ord=2, dim=-1), min=0.0).mean()

        loss_tuple = (loss_v1, loss_v2, loss_v3, loss_d, loss_reg1, loss_reg2, loss_con, loss_sd)
        loss_sum = sum(loss_tuple)
        return loss_sum, loss_tuple

