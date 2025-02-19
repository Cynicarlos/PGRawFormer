import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath, trunc_normal_

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )
    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )
    def forward(self, x):
        return self.body(x)

class ChannelRemainUp(nn.Module):
    def __init__(self, in_channels):
        super(ChannelRemainUp, self).__init__()
        self.body = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.body(x)

class SimpleFuse(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Conv2d(2*in_channels, in_channels, 1)
    def forward(self, x, y):
        return self.body(torch.cat([x,y],dim=1))

class IlluminationEstimator(nn.Module):
    def __init__(self, in_channels, dim):
        super(IlluminationEstimator, self).__init__()
        self.dim = dim
        self.conv1 = nn.Conv2d(in_channels+1, dim, 3, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, in_channels, 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        x = torch.cat([x, x.mean(dim=1).unsqueeze(1)],dim=1) #(b, in_c + 1, h, w)
        x = self.conv1(x)
        l = self.conv2(x) #(b, in_c, h, w)
        r = self.conv3(x) #(b, c, h, w)
        return l, r

class PFuse(nn.Module):
    def __init__(self, dim, num_meta_keys):
        super(PFuse, self).__init__()
        self.meta_fc1 = nn.Linear(num_meta_keys, num_meta_keys)
        self.meta_fc2 = nn.Linear(num_meta_keys, dim)
        self.conv_in = nn.Conv2d(dim, 2*dim, 1)
        self.conv1 = nn.Conv2d(dim+num_meta_keys, dim, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*dim, 2*dim, 3, padding=1, groups=dim),
            nn.Conv2d(2*dim, dim, 1)
        )
        
    def forward(self, x, metainfo):
        _,_,h,w = x.shape
        shortcut = x
        x, x1 = self.conv_in(x).chunk(2, dim=1)
        x = self.conv1(torch.cat([x, self.meta_fc1(metainfo).unsqueeze(-1).unsqueeze(-1).expand(-1,-1,h,w)], dim=1))
        x1 = x1 * self.meta_fc2(metainfo).unsqueeze(-1).unsqueeze(-1)
        x = torch.cat([x, x1], dim=1)
        x = self.conv2(x)
        x = x + shortcut
        return x

class PGFFN(nn.Module):
    def __init__(self, dim, num_meta_keys, ffn_expansion_factor, bias):
        super(PGFFN, self).__init__()
        self.pfuse = PFuse(dim=dim, num_meta_keys=num_meta_keys)

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x, metainfo):
        x = self.pfuse(x, metainfo)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class PGCSA(nn.Module):
    def __init__(self, dim, in_channels, num_heads, bias):
        super(PGCSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.q = nn.Linear(dim, dim//num_heads, bias=False)
        self.k = nn.Linear(dim, dim//num_heads, bias=False)
        self.v1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1)
        )
        self.v2 = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(2*dim, dim)
        
        self.refine_l = nn.Linear(in_channels, dim)
        self.refine_r = nn.Linear(dim, dim)
        
        self.linear_out = nn.Linear(dim//num_heads, dim//num_heads, bias=True)
        self.conv_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, l, r):
        #x, r (b,c,h,w)
        #l (b,in_c,h,w)
        b,c,h,w = x.shape
        l = rearrange(l, 'b c h w -> b (h w) c')
        r = rearrange(r, 'b c h w -> b (h w) c')
        
        v1 = self.v1(x) #(b,c,h,w)
        v1 = rearrange(v1, 'b c h w -> b (h w) c')
        
        x = rearrange(x, 'b c h w -> b (h w) c')
        q = self.q(x)
        k = self.k(x)
        v2 = self.v2(x) # (b, hw, c)
        v2 = self.refine_l(l)*v2 + self.refine_r(r)
        v = self.v(torch.cat([v1, v2], dim=-1))
        
        q = rearrange(q, 'b l (head c) -> b head c l', head=1)#(b, 1, c//mum_heads, hw)
        k = rearrange(k, 'b l (head c) -> b head c l', head=1)#(b, 1, c//mum_heads, hw)
        v = rearrange(v, 'b l (head c) -> b head c l', head=self.num_heads)#(b, mum_heads, c//mum_heads, hw)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature # (b, 1, c//mum_heads, c//mum_heads)
        attn = attn.softmax(dim=-1)
        attn = attn.repeat(1,self.num_heads,1,1)
        x = (attn @ v)# (b, mum_heads, c//mum_heads, hw)
        x = self.linear_out(x.permute(0,1,3,2))# (b, mum_heads, hw, c//mum_heads)
        x = rearrange(x, 'b head (h w) c -> b (head c) h w', h=h, w=w)
        x = self.conv_out(x)
        return x

class PGFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, in_channels, num_meta_keys, ffn_expansion_factor, bias, LayerNorm_type):
        super(PGFormerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.csa = PGCSA(dim, in_channels, num_heads, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = PGFFN(dim, num_meta_keys, ffn_expansion_factor, bias)

    def forward(self, x, l, r, metainfo):
        x= x + self.csa(self.norm1(x), l, r)
        x = x + self.ffn(self.norm2(x), metainfo)
        return x

from utils.registry import MODEL_REGISTRY
@MODEL_REGISTRY.register()
class PGRawFormer(nn.Module):
    def __init__(self, 
        in_channels=4, 
        out_channels=4, 
        dim=32,
        layers=4,
        num_meta_keys=4,
        num_blocks=[1,1,1,1], 
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias'
    ):

        super(PGRawFormer, self).__init__()
        assert len(num_blocks) == layers and len(heads) == layers
        
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, padding=1)
        )
        
        self.ie = IlluminationEstimator(in_channels=in_channels, dim=dim)

        self.encoders = nn.ModuleList([
            nn.ModuleList([
                PGFormerBlock(int(dim*2**i), heads[i], in_channels, num_meta_keys, ffn_expansion_factor, bias, LayerNorm_type)
                for _ in range(num_blocks[i])
            ])
            for i in range (layers-1)
        ])
        
        self.middle_block = nn.ModuleList([
            PGFormerBlock(int(dim*2**(layers-1)), heads[layers-1], in_channels, num_meta_keys, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[layers-1])
        ])
        
        self.decoders = nn.ModuleList([
            nn.ModuleList([
                PGFormerBlock(int(dim*2**i), heads[i], in_channels, num_meta_keys, ffn_expansion_factor, bias, LayerNorm_type)
                for _ in range(num_blocks[i])
            ])
            for i in range (layers-2, -1, -1)
        ])
        
        self.downs = nn.ModuleList([
            Downsample(n_feat=dim*2**i)
            for i in range (layers-1)
        ])
        
        self.l_down = nn.AvgPool2d(2, 2)
        
        self.r_downs = nn.ModuleList([
            Downsample(n_feat=dim*2**i)
            for i in range (layers-1)
        ])

        self.ups = nn.ModuleList([
            Upsample(n_feat=dim*2**i)
            for i in range (layers-1, -1, -1) 
        ])

        self.fuses = nn.ModuleList([
            SimpleFuse(in_channels=dim*2**i)
            for i in range (layers-2, -1, -1)
        ])
        
        self.output = nn.Sequential( 
            nn.Conv2d(dim, in_channels, 3, padding=1)
        )

    def forward(self, x, metainfo):
        #matainfo : (b,n)
        x = self._check_and_padding(x)
        shortcut = x
        
        l, r = self.ie(x) #(b, c_in, h, w) (b, c, h, w)
        x = self.conv_in(x)
        
        ls = []
        rs = []
        for r_down in self.r_downs:
            ls.append(l)
            l = self.l_down(l)
            rs.append(r)
            r = r_down(r)

        encode_features = []
        for encodes, _l, _r, down in zip(self.encoders, ls, rs, self.downs):
            for encode in encodes:
                x = encode(x, _l, _r, metainfo)
            encode_features.append(x)
            x = down(x)

        for block in self.middle_block:
            x = block(x, l, r, metainfo)
        
        encode_features.reverse()
        ls.reverse()
        rs.reverse()
        for up, fuse, feature, decodes, l, r in zip(
            self.ups, self.fuses, encode_features, self.decoders, ls, rs
        ):
            x = up(x)
            x = fuse(x, feature)
            for decode in decodes:
                x = decode(x, l, r, metainfo)
        x = self.output(x)
        x = x + shortcut
        x = self._check_and_crop(x)
        return x
    
    def _check_and_padding(self, x):
        _, _, h, w = x.size()
        stride = 2**3
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

def cal_model_complexity(model, x, metainfo):
    import thop
    flops, params = thop.profile(model, inputs=(x, metainfo,), verbose=False)
    print(f"FLOPs: {flops / 1e9} G")
    print(f"Params: {params / 1e6} M")


if __name__ == '__main__':
    from tqdm import tqdm
    model = PGRawFormer(in_channels=4, out_channels=4, dim=32, layers=4,num_meta_keys=4,
                        num_blocks=[2,2,2,2], heads=[1, 2, 4, 8]).cuda()

    iterations = 10

    random_input = torch.randn(1, 4, 1024, 1024).cuda()
    metainfo = torch.randn(1, 4).cuda()
    cal_model_complexity(model, random_input, metainfo)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    times = torch.zeros(iterations)

    with torch.no_grad():
        for _ in range(10):
            _ = model(random_input, metainfo)

    with torch.no_grad():
        for iter in tqdm(range(iterations), desc="Measuring Inference Time", unit="iteration"):
            starter.record()
            _ = model(random_input, metainfo)
            ender.record()

            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            times[iter] = curr_time

    mean_time = times.mean().item()
    fps = 1000 / mean_time

    print(f"Inference time: {mean_time:.6f} ms, FPS: {fps:.2f}")