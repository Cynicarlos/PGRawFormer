import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision import transforms
from pdb import set_trace as stx

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

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
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

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, num_meta_keys=5, meta_embedding_dims=80):
        super(Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        meta_dims = num_meta_keys*meta_embedding_dims
        sqrt_q_dim = int(math.sqrt(meta_dims))
        self.resize = transforms.Resize([sqrt_q_dim, sqrt_q_dim],antialias=True)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Linear(meta_dims, meta_dims, bias=bias)
        
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x, metainfo):
        b,c,h,w = x.shape

        q = self.q(metainfo.flatten(1)) #(b,nd)
        k, v = self.kv_dwconv(self.kv(x)).chunk(2, dim=1) #(b,c,h,w)
        k = self.resize(k) # (b,c,sqrt(nd),sqrt(nd))
	
        q = repeat(q, 'b l -> b head c l', head=self.num_heads, c=self.dim//self.num_heads) # (b, num_heads, head_dims, nd)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # (b, num_heads, head_dims, nd)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # (b, num_heads, head_dims, hw)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature # (b, num_heads, head_dims, head_dims)
        attn = attn.softmax(dim=-1)

        out = (attn @ v) # (b, num_heads, head_dims, hw)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, num_meta_keys, meta_embedding_dims, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, num_meta_keys, meta_embedding_dims)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, metainfo):
        x = x + self.attn(self.norm1(x), metainfo)
        x = x + self.ffn(self.norm2(x))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=32, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

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

from utils.registry import MODEL_REGISTRY
@MODEL_REGISTRY.register()
class MetaRawFormer(nn.Module):
    def __init__(self, 
        in_channels=4, 
        out_channels=4, 
        dim=32,
        num_meta_keys=5,
        meta_table_size=[30,26,26,250,3],
        meta_embedding_dims=80,
        num_blocks=[1,1,2,2], 
        num_refinement_blocks=4,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias'
    ):

        super(MetaRawFormer, self).__init__()

        self.num_meta_keys = num_meta_keys
        self.meta_embeddings = nn.ModuleList([
            nn.Embedding(size, meta_embedding_dims) for size in meta_table_size
        ])
        
        self.patch_embed = OverlapPatchEmbed(in_channels, dim)

        self.encoders = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(int(dim*2**i), heads[i], num_meta_keys, meta_embedding_dims, ffn_expansion_factor, bias, LayerNorm_type) 
                for _ in range(num_blocks[i])
            ])
            for i in range (3)
        ])
        
        self.middle_block = nn.ModuleList([
            TransformerBlock(int(dim*2**3), heads[3], num_meta_keys, meta_embedding_dims, ffn_expansion_factor, bias, LayerNorm_type) 
            for _ in range(num_blocks[3])
        ])
        
        self.decoders = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(int(dim*2**i), heads[i], num_meta_keys, meta_embedding_dims, ffn_expansion_factor, bias, LayerNorm_type) 
                for _ in range(num_blocks[i])
            ])
            for i in range (2, 0, -1)
        ])
        
        self.last_decoder = nn.ModuleList([
            TransformerBlock(int(dim*2**1), heads[0], num_meta_keys, meta_embedding_dims, ffn_expansion_factor, bias, LayerNorm_type) 
            for _ in range(num_blocks[0])
        ])
        
        self.downs = nn.ModuleList([
            Downsample(n_feat=dim*2**i)
            for i in range (3)
        ])
        
        self.ups = nn.ModuleList([
            Upsample(n_feat=dim*2**i)
            for i in range (3, 0, -1)
        ])
        self.fuses = nn.ModuleList([
            SimpleFuse(in_channels=dim*2**i)
            for i in range (2, 0, -1)
        ])

        self.refinement = nn.ModuleList([
            TransformerBlock(dim*2**1, heads[0], num_meta_keys, meta_embedding_dims, ffn_expansion_factor, bias, LayerNorm_type) 
            for i in range(num_refinement_blocks)
        ])
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x, metainfoidx):
        metainfo = []
        for i in range(self.num_meta_keys):
            embedding = self.meta_embeddings[i](metainfoidx[:, i])
            metainfo.append(embedding)
        metainfo = torch.stack(metainfo,dim=1) #(b,n,d)
        
        shortcut = x
        x = self._check_and_padding(x)
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
        for up, fuse, feature, decodes in zip(self.ups[:-1], self.fuses, encode_features[:-1], self.decoders):
            x = up(x)
            x = fuse(x, feature)
            for decode in decodes:
                x = decode(x, metainfo)
        
        x = self.ups[-1](x)
        x = torch.cat([x, encode_features[-1]],dim=1)
        for last_block in self.last_decoder:
            x = last_block(x, metainfo)

        for refine_block in self.refinement:
            x = refine_block(x, metainfo)
        x = self.output(x)
        x = self._check_and_crop(x)
        x = x + shortcut

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
    
def cal_model_complexity(model, x, metainfoidx):
    import thop
    flops, params = thop.profile(model, inputs=(x, metainfoidx,), verbose=False)
    print(f"FLOPs: {flops / 1e9} G")
    print(f"Params: {params / 1e6} M")

if __name__ == '__main__':
    model = MetaRawFormer(in_channels=4, out_channels=4, dim=32, 
                        num_meta_keys=5, meta_table_size=[30,26,26,250,3], meta_embedding_dims=80,
                        num_blocks=[1, 1, 2, 2], num_refinement_blocks=4, heads=[1, 2, 4, 8]).cuda()
    idx_list = [1,2,3,4,2]
    metainfoidx = torch.tensor(idx_list, dtype=torch.long).unsqueeze(0).cuda()
    x = torch.rand(1,4,512,512).cuda()
    cal_model_complexity(model, x, metainfoidx)
    exit(0)
    import time
    begin = time.time()
    x = model(x)
    end = time.time()
    print(f'Time comsumed: {end-begin} s')
    
'''
FLOPs: 103.061359072 G
Params: 7.149408 M
'''