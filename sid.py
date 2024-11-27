import torch
import torch.nn as nn
import torch.nn.functional as F
class DoubleConv(nn.Module):
    #  Conv -> BN -> LReLU -> Conv -> BN -> LReLU
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),)
    def forward(self, x):
        x = self.f(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.f = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),)
    def forward(self, x):
        x = self.f(x)
        return x


class Up(nn.Module):
    # upsample and concat
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.f = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        x = self.f(x)
        return x

from utils.registry import MODEL_REGISTRY
@MODEL_REGISTRY.register()
class SID(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.d1 = Down(32, 64)
        self.d2 = Down(64, 128)
        self.d3 = Down(128, 256)
        self.d4 = Down(256, 512)

        self.u1 = Up(512, 256)
        self.u2 = Up(256, 128)
        self.u3 = Up(128, 64)
        self.u4 = Up(64, 32)
        self.outc = OutConv(32, in_channels)

    def forward(self, x):
        x = self._check_and_padding(x)
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        x = self.outc(x)
        x = self._check_and_crop(x)
        return x

    def _check_and_padding(self, x):
        _, _, h, w = x.size()
        stride = 2**4
        dh = -h % stride
        dw = -w % stride
        top_pad = dh // 2
        bottom_pad = dh - top_pad
        left_pad = dw // 2
        right_pad = dw - left_pad
        self.crop_indices = (left_pad, w + left_pad, top_pad, h + top_pad)

        # Pad the tensor with reflect mode
        padded_tensor = F.pad(
            x, (left_pad, right_pad, top_pad, bottom_pad), mode="reflect"
        )

        return padded_tensor

    def _check_and_crop(self, x):
        left, right, top, bottom = self.crop_indices
        x = x[:, :, top:bottom, left:right]
        return x

def cal_model_complexity(model, x):
    import thop
    flops, params = thop.profile(model, inputs=(x,), verbose=False)
    print(f"FLOPs: {flops / 1e9} G")
    print(f"Params: {params / 1e6} M")

if __name__ == '__main__':
    model = SID().cuda()
    x = torch.rand(1,4,256,256).cuda()
    cal_model_complexity(model, x)
    #exit(0)
    import time
    begin = time.time()
    x = model(x)
    print(x.shape)
    end = time.time()
    print(f'Time comsumed: {end-begin} s')