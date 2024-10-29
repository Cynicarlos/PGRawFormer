import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss

class PSNRLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.scale = 10 / np.log(10)
        #self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        #self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        return  self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        #self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(4, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)    # filter
        down = filtered[:, :, ::2, ::2]               # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down*4                  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

class Losses(nn.Module):
    def __init__(self, types, weights):
        super().__init__()
        self.module_list = nn.ModuleList()
        self.types = types
        self.weights = weights
        for loss_type in types:
            if loss_type == 'MSE':
                self.module_list.append(MSELoss())
            elif loss_type == 'L1':
                self.module_list.append(L1Loss())
            elif loss_type == 'PSNR':
                self.module_list.append(PSNRLoss())
            elif loss_type == 'Charbonnier':
                self.module_list.append(CharbonnierLoss())
            elif loss_type == 'Edge':
                self.module_list.append(EdgeLoss())

    def __len__(self):
        return len(self.types)

    def forward(self, preds, gts):
        losses = []
        for i in range(len(self.types)):
            loss = self.module_list[i](preds[i],gts[i]) * self.weights[i]
            losses.append(loss)
        return losses

def build_loss(config):
    loss_types = config['types']
    loss_weights = config['weights']

    assert len(loss_weights) == len(loss_types)
    criterion = Losses(types=loss_types, weights=loss_weights)
    return criterion
