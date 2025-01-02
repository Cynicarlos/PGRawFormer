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

class SimpleFuse(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Conv2d(2*in_channels, in_channels, 1)
    def forward(self, x, y):
        return self.body(torch.cat([x,y],dim=1))

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.conv_r = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU()
        )
        self.conv_l = nn.Sequential(
            nn.Conv2d(4, 4, 1),
            nn.Conv2d(4, 4, 3, padding=1),
            nn.GELU()
        )
        self.project_in = nn.Conv2d(2*dim+4, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x, l, r):
        #x,r (b,c,h,w)
        #l (b,4,h,w)
        x = torch.cat([x, self.conv_r(r), self.conv_l(l)], dim=1)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature # (b, head, c, c)
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class MetaIlluminationEstimator(nn.Module):
    def __init__(self, in_channels, dim, num_meta_keys, meta_dims):
        super(MetaIlluminationEstimator, self).__init__()
        self.dim = dim
        self.conv_in = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fcs = nn.Sequential(
            nn.Linear(2*dim, dim),
            nn.GELU(),
            nn.Linear(dim, 2*dim)
        )
        
        self.meta_fc = nn.Linear(num_meta_keys*meta_dims, dim)
        
        self.conv1 = nn.Conv2d(dim+1, dim, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, bias=True, groups=in_channels)

        self.conv2 = nn.Conv2d(dim, in_channels, kernel_size=1, bias=True)

    def forward(self, x, metainfo):
        mean_c = x.mean(dim=1).unsqueeze(1) #(b, 1, h, w)
        x = self.conv_in(x) #(b,c,h,w)
        metainfo = self.meta_fc(metainfo.flatten(1))#(b, nd) -> (b, c)
        squeezed_x = self.squeeze(x).view(-1, self.dim)#(b, c)
        squeezed_x = torch.cat([squeezed_x, metainfo], dim=1) #(b, 2c)
        w, bias = torch.chunk(self.fcs(squeezed_x), chunks=2, dim=1) #2 (b, c)
        w = w.view(-1,self.dim,1,1)
        bias = bias.view(-1,self.dim,1,1)
        x = w*x + bias

        x = torch.cat([x,mean_c], dim=1) #(b, in_c + 1, h, w)
        x_1 = self.conv1(x)
        r = self.depth_conv(x_1) #(b, c, h, w)
        l = self.conv2(r) #(b, in_c, h, w)
        return r, l

class MetaAttention(nn.Module):
    def __init__(self, dim, num_heads, num_meta_keys, meta_dims, bias):
        super(MetaAttention, self).__init__()
        self.num_heads = num_heads
        self.num_meta_keys = num_meta_keys
        self.meta_dims = meta_dims

        self.conv_in = nn.Conv2d(dim, 2*dim, 1)
        self.fc = nn.Linear(num_meta_keys*meta_dims, num_meta_keys)
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.qkv = nn.Conv2d(dim+num_meta_keys, dim+2*(dim//num_heads), kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim+2*(dim//num_heads), dim+2*(dim//num_heads), kernel_size=3, stride=1, padding=1, groups=dim+2*(dim//num_heads), bias=bias)
        
        self.conv_l = nn.Sequential(
            nn.Conv2d(4, dim, 1),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU()
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, l, metainfo):
        #x (b,c,h,w)
        #l (b,4,h,w)
        #metainfo (b,n,d)
        b,c,h,w = x.shape
        _,n,d = metainfo.shape
        
        metainfo = self.fc(metainfo.flatten(1))#(b, nd) -> (b, n)
        metainfo = metainfo.unsqueeze(-1).unsqueeze(-1).expand(b,n,h,w)
        
        qkv = self.qkv_dwconv(self.qkv(torch.cat([x, metainfo], dim=1)))
        q,k,v = qkv[:,0:c//self.num_heads,:,:], qkv[:,c//self.num_heads:2*(c//self.num_heads),:,:], qkv[:,2*(c//self.num_heads):,:,:]

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=1)#(b, 1, c//n_head, hw)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=1)#(b, 1, c//n_head, hw)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature # (b, 1, c//head, c//head)
        attn = attn.softmax(dim=-1)#(b, 1, c//n_head, c//n_head)
        
        l = self.conv_l(l)
        v = l*v

        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        x = [attn @ v[:, i, :, :].unsqueeze(1) for i in range(self.num_heads)] # n_head (b, 1, c//head, hw)
        x = torch.cat(x, dim=1)#(b, n_head, c//n_head, hw)
        x = rearrange(x, 'b head c (h w) -> b (head c) h w', h=h, w=w)
        x = self.project_out(x)
        return x

class MetaTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, num_meta_keys, meta_dims, ffn_expansion_factor, bias, LayerNorm_type):
        super(MetaTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.meta_attn = MetaAttention(dim, num_heads, num_meta_keys, meta_dims, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, metainfo, l, r):
        x = x + self.meta_attn(self.norm1(x), l, metainfo)
        x = x + self.ffn(self.norm2(x), l, r)
        return x


class RemainChannelsDown(nn.Module):
    def __init__(self, in_channels):
        super(RemainChannelsDown, self).__init__()
        self.body = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)
    def forward(self, x):
        return self.body(x)

from utils.registry import MODEL_REGISTRY
@MODEL_REGISTRY.register()
class MetaRawFormer(nn.Module):
    def __init__(self, 
        in_channels=4, 
        out_channels=4, 
        dim=32,
        layers=4,
        num_meta_keys=4,
        meta_dims=32,
        num_blocks=[2,2,2,4], 
        num_refinement_blocks=2,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias'
    ):

        super(MetaRawFormer, self).__init__()
        assert len(num_blocks) == layers and len(heads) == layers

        self.num_meta_keys = num_meta_keys
        self.meta_dims = meta_dims
        self.meta_project = nn.Linear(num_meta_keys, num_meta_keys*meta_dims)

        self.in_conv = nn.Conv2d(in_channels, dim, 3, padding=1)

        self.mie = MetaIlluminationEstimator(in_channels=in_channels, dim=dim, num_meta_keys=num_meta_keys, meta_dims=meta_dims)

        self.encoders = nn.ModuleList([
            nn.ModuleList([
                MetaTransformerBlock(int(dim*2**i), heads[i], num_meta_keys, meta_dims, ffn_expansion_factor, bias, LayerNorm_type)
                for _ in range(num_blocks[i])
            ])
            for i in range (layers-1)
        ])
        
        self.middle_block = nn.ModuleList([
            MetaTransformerBlock(int(dim*2**(layers-1)), heads[layers-1], num_meta_keys, meta_dims, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[layers-1])
        ])
        
        self.decoders = nn.ModuleList([
            nn.ModuleList([
                MetaTransformerBlock(int(dim*2**i), heads[i], num_meta_keys, meta_dims, ffn_expansion_factor, bias, LayerNorm_type)
                for _ in range(num_blocks[i])
            ])
            for i in range (layers-2, -1, -1)
        ])

        self.downs = nn.ModuleList([
            Downsample(n_feat=dim*2**i)
            for i in range (layers-1)
        ])
        
        self.l_downs = nn.ModuleList([
            RemainChannelsDown(in_channels=4)
            for i in range (layers-1)
        ])
        
        self.r_downs = nn.ModuleList([
            Downsample(n_feat=dim*2**i)
            for i in range (layers-1)
        ])

        self.ups = nn.ModuleList([
            Upsample(n_feat=dim*2**i)
            for i in range (layers-1, 0, -1) 
        ])

        self.fuses = nn.ModuleList([
            SimpleFuse(in_channels=dim*2**i)
            for i in range (layers-2, -1, -1)
        ])
        
        self.refine_conv = nn.Sequential(
            nn.Conv2d(dim, 2*dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(2*dim, 2*dim, 1),
        )
        
        self.refine_conv_r = nn.Sequential(
            nn.Conv2d(dim, 2*dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(2*dim, 2*dim, 1),
        )

        self.refinement = nn.ModuleList([
            MetaTransformerBlock(2*dim, heads[0], num_meta_keys, meta_dims, ffn_expansion_factor, bias, LayerNorm_type)
            for i in range(num_refinement_blocks)
        ])

        self.output = nn.Sequential(
            nn.Conv2d(2*dim, 2*dim, 3, padding=1, groups=dim),
            ChannelAttention(num_feat=2*dim),
            nn.Conv2d(2*dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, out_channels, 1)
        )

    def forward(self, x, metainfo):
        #matainfo : (b,n)
        #mask (b,1,h,w)
        x = self._check_and_padding(x)
        shortcut = x
        metainfo = self.meta_project(metainfo).view(-1, self.num_meta_keys, self.meta_dims)#(b,n,d)
        
        r, l = self.mie(x, metainfo) #(b,c,h,w) (b,4,h,w)
        x = l * x #(b,4,h,w)

        x = self.in_conv(x)#c

        ls = []
        rs = []
        for l_down, r_down in zip(self.l_downs, self.r_downs):
            ls.append(l)
            l = l_down(l)
            rs.append(r)
            r = r_down(r)

        encode_features = []
        for encodes, _l, _r, down in zip(self.encoders, ls, rs, self.downs):
            for encode in encodes:
                x = encode(x, metainfo, _l, _r)
            encode_features.append(x)
            x = down(x)

        for block in self.middle_block:
            x = block(x, metainfo, l, r)
        
        encode_features.reverse()
        ls.reverse()
        rs.reverse()
        for up, fuse, feature, decodes, l, r in zip(
            self.ups, self.fuses, encode_features, self.decoders, ls, rs
        ):
            x = up(x)
            x = fuse(x, feature)
            for decode in decodes:
                x = decode(x, metainfo, l, r)

        x = self.refine_conv(x)
        r = self.refine_conv_r(rs[-1])
        for refine_block in self.refinement:
            x = refine_block(x, metainfo, ls[-1], r)

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
    model = MetaRawFormer(in_channels=4, out_channels=4, dim=32, layers=4,
                        num_meta_keys=4, meta_dims=32, 
                        num_blocks=[2,2,2,4], num_refinement_blocks=2, heads=[1, 2, 4, 8]).cuda()
    metainfo = torch.rand((1,4)).cuda()
    x = torch.rand(1,4,1024,1024).cuda()
    with torch.no_grad():
        cal_model_complexity(model, x, metainfo)
        #exit(0)
        import time
        begin = time.time()
        x = model(x, metainfo)
        end = time.time()
        print(f'Time comsumed: {end-begin} s')