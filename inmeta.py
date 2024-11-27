import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
from timm.models.vision_transformer import Mlp

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

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=32):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.proj(x)

        return x

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
'''
class MetaAwareFuse(nn.Module):
    def __init__(self, in_channels, meta_dims):
        super(MetaAwareFuse, self).__init__()
        self.fc    = nn.Linear(meta_dims, in_channels)
        self.beta  = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        #self.ffn = FeedForward(dim=in_channels, ffn_expansion_factor=2, bias=True)
        self.ffn = NAFBlock(c=in_channels)
    def forward(self, x, metainfo):
        #metainfo: (b, d)
        shortcut = x
        gating_factors = torch.sigmoid(self.fc(metainfo))#(b,c)
        gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1)
        x = x * self.gamma + self.beta  # 1) learned feature scaling/modulation
        x = x * gating_factors          # 2) (soft) feature routing based on text
        x = self.ffn(x)               # 3) block feature enhancement
        x = x + shortcut
        return x
'''

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
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

class MetaAttention(nn.Module):
    def __init__(self, dim, num_heads, num_meta_keys, meta_embedding_dims, bias):
        super(MetaAttention, self).__init__()
        self.num_heads = num_heads
        self.num_meta_keys = num_meta_keys
        self.meta_embedding_dims = meta_embedding_dims
        
        self.fc = nn.Linear(num_meta_keys*meta_embedding_dims, num_meta_keys)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim+num_meta_keys, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, metainfo):
        #x (b,c,h,w)
        #metainfo (b,n,d)
        b,c,h,w = x.shape
        _,n,d = metainfo.shape

        metainfo = self.fc(metainfo.flatten(1))#(b, nd) -> (b, n)
        metainfo = metainfo.unsqueeze(-1).unsqueeze(-1).expand(b,n,h,w)
        
        x = torch.cat([x, metainfo], dim=1) #(b, c+n, h, w)
        
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

class ParallelMetaAttention(nn.Module):
    def __init__(self, dim, num_heads, num_meta_keys, meta_dims):
        super(ParallelMetaAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_meta_keys = num_meta_keys
        self.meta_dims = meta_dims
        
        self.qs_proj = nn.ModuleList([
            nn.Linear(meta_dims, dim)
            for _ in range (num_meta_keys)
        ])
        
        self.temperatures = [
            nn.Parameter(torch.ones(num_heads, 1, 1)).to('cuda')
            for _ in range (num_meta_keys)
        ]
        
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2)
        
        self.cross_conv = nn.Sequential(
            nn.Conv2d(dim//num_heads, dim, 3, padding=1, groups=dim//num_heads),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x, metainfo):
        #x (b,c,h,w)
        #metainfo (b,n,d)
        b,c,h,w = x.shape
        _,n,d = metainfo.shape
        
        qs = list(metainfo.chunk(n, dim=1)) #n (b,1,d)
        qs = [q_proj(q).view(b,c,1) for (q, q_proj) in zip (qs, self.qs_proj)]#n (b,c,1)

        qs = [rearrange(q, 'b (head c) n -> b head c n', head=self.num_heads) for q in qs]
        qs = [torch.nn.functional.normalize(q, dim=-1) for q in qs]#n (b,head,c//head,1)
        
        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1) #(b,c,h,w)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = torch.nn.functional.normalize(k, dim=-1)#(b,head,c//head,hw)

        cross_attens = [(q.transpose(-2, -1) @ k) * temp for (q, temp) in zip(qs, self.temperatures)]#n (b,head,1,hw)
        cross_attens = [attn.softmax(dim=-1) for attn in cross_attens]
        cross_attens = [rearrange(cross_atten, 'b head n (h w) -> b h w n head', h=h, w=w) for cross_atten in cross_attens]#n (b, h, w, 1, head)
        
        v = rearrange(v, 'b (head c) h w -> b h w head c', head=self.num_heads)#(b, h, w, head, c//head)
        out = [(cross_atten @ v) for cross_atten in cross_attens]#n (b, h, w, 1, c//head)
        out = [rearrange(o, 'b h w n c -> b (n c) h w') for o in out]#n (b, c//head, h, w)
        
        # out = torch.cat(out, dim=1)#(b, nc//head, h, w)
        out = torch.sum(torch.stack(out), dim=0)# (b, c//head, h, w)
        out = self.cross_conv(out) #(b c h w)

        return out

class TransformerBlcok(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlcok, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class MetaTransformerBlcok(nn.Module):
    def __init__(self, dim, num_heads, num_meta_keys, meta_dims, ffn_expansion_factor, bias, LayerNorm_type):
        super(MetaTransformerBlcok, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.meta_attn = MetaAttention(dim, num_heads, num_meta_keys, meta_dims, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, metainfo):
        x = x + self.meta_attn(self.norm1(x), metainfo)
        x = x + self.ffn(self.norm2(x))
        return x

class ResBlcok(nn.Module):
    def __init__(self, block):
        super(ResBlcok, self).__init__()
        self.block = block
    def forward(self, x):
        return x + self.block(x)

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
        num_blocks=[4,6,6,8], 
        num_refinement_blocks=4,
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

        self.patch_embed = OverlapPatchEmbed(in_channels, dim)

        self.encoders = nn.ModuleList([
            nn.ModuleList([
                #TransformerBlcok(int(dim*2**i), heads[i], ffn_expansion_factor, bias, LayerNorm_type)
                MetaTransformerBlcok(int(dim*2**i), heads[i], num_meta_keys, meta_dims, ffn_expansion_factor, bias, LayerNorm_type)
                for _ in range(num_blocks[i])
            ])
            for i in range (layers-1)
        ])
        
        self.middle_block = nn.ModuleList([
            #TransformerBlcok(int(dim*2**(layers-1)), heads[layers-1], ffn_expansion_factor, bias, LayerNorm_type)
            MetaTransformerBlcok(int(dim*2**(layers-1)), heads[layers-1], num_meta_keys, meta_dims, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[layers-1])
        ])
        
        self.decoders = nn.ModuleList([
            nn.ModuleList([
                #TransformerBlcok(int(dim*2**i), heads[i], ffn_expansion_factor, bias, LayerNorm_type)
                MetaTransformerBlcok(int(dim*2**i), heads[i], num_meta_keys, meta_dims, ffn_expansion_factor, bias, LayerNorm_type)
                for _ in range(num_blocks[i])
            ])
            for i in range (layers-2, -1, -1)
        ])

        self.downs = nn.ModuleList([
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

        self.refinement = nn.ModuleList([
            #TransformerBlcok(2*dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
            MetaTransformerBlcok(2*dim, heads[0], num_meta_keys, meta_dims, ffn_expansion_factor, bias, LayerNorm_type)
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
        x = self._check_and_padding(x)
        shortcut = x
        metainfo = self.meta_project(metainfo).view(-1, self.num_meta_keys, self.meta_dims)#(b,n,d)

        x = self.patch_embed(x)#c

        encode_features = []
        for encodes, down in zip(self.encoders, self.downs):
            for encode in encodes:
                x = encode(x, metainfo)
            encode_features.append(x)
            x = down(x)

        for block in self.middle_block:
            x = block(x, metainfo)
        
        encode_features.reverse()
        
        for up, fuse, feature, decodes in zip(
            self.ups, self.fuses, encode_features, self.decoders
        ):
            x = up(x)
            x = fuse(x, feature)
            for decode in decodes:
                x = decode(x, metainfo)

        x = self.refine_conv(x)
        for refine_block in self.refinement:
            x = refine_block(x, metainfo)

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
                        num_meta_keys=4, meta_dims=32, num_blocks=[2,2,2,2], num_refinement_blocks=2, heads=[1, 2, 4, 8]).cuda()
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