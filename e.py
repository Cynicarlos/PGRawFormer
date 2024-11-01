## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange

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

class MetaAttention(nn.Module):
    def __init__(self, dim, num_heads, num_meta_keys, meta_embedding_dims):
        super(MetaAttention, self).__init__()
        self.fc    = nn.Linear(num_meta_keys*meta_embedding_dims, dim)
        self.conv  = nn.Conv2d(dim, 2*dim, 3, padding=1, groups=dim)
        self.beta  = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        #self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x, metainfo):
        #x (b,c,h,w)
        #metainfo (b,n,d)
        b,c,h,w = x.shape
        metainfo = metainfo.flatten(1)#(b, nd)
        metainfo = self.fc(metainfo)#(b, c)
        metainfo = metainfo.unsqueeze(-1).unsqueeze(-1) #(b, c, 1, ,1)
        #metainfo = metainfo.unsqueeze(-1).unsqueeze(-1).expand(x.shape) #(b, c, h, w)
        #metainfo = self.conv(metainfo) #2c
        #s,t = torch.chunk(metainfo, 2, dim=1)
        #q = self.q(self.beta * s * x + self.gamma * t)
        q = self.q(self.beta * metainfo * x)
        kv = self.kv_dwconv(self.kv(x))
        k,v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

'''
class MetaInjection(nn.Module):
    def __init__(self, in_channels, num_meta_keys, meta_embedding_dims, num_heads):
        super(MetaInjection, self).__init__()
        self.fc    = nn.Linear(num_meta_keys*meta_embedding_dims, in_channels)
        self.meta_attention = MetaAttention(dim=in_channels, num_heads=num_heads)
        self.beta  = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)

    def forward(self, x, metainfo):
        #metainfo (b,n,d)
        shortcut = x
        metainfo = metainfo.flatten(1)#(b, nd)
        metainfo = self.fc(metainfo)#(b, c)
        metainfo = metainfo.unsqueeze(-1).unsqueeze(-1).expand(x.shape) #(b, c, h, w)
        x = self.meta_attention(x, metainfo)
        return x + shortcut
'''

class SimpleFuse(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Conv2d(2*in_channels, in_channels, 1)
    def forward(self, x, y):
        return self.body(torch.cat([x,y],dim=1))

class MetaAwareFuse(nn.Module):
    def __init__(self, in_channels, num_meta_keys, meta_embedding_dims):
        super().__init__()
        self.fc = nn.Linear(num_meta_keys*meta_embedding_dims, in_channels)
        self.alpha = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, 1),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, 1),
            nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*in_channels, 2*in_channels, 1),
            nn.GELU()
        )
    def forward(self, x, y, metainfo):
        #x, y (b,c,h,w)
        #metainfo (b,n,d)
        b,c,h,w = x.shape
        metainfo = metainfo.flatten(1)#(b, nd)
        metainfo = self.fc(metainfo)#(b, c)
        metainfo = metainfo.unsqueeze(-1).unsqueeze(-1) #(b, c, 1, 1)
        t = torch.cat([x,y], dim=1) #2c
        x = self.alpha * metainfo * x 
        y = self.beta * metainfo * y
        x = torch.cat([x,y], dim=1)
        x = self.conv1(x) #c
        t = self.conv2(t) #c
        x = torch.cat([x,t],dim=1)
        x = self.conv3(x) #2c
        x, t = torch.chunk(x, 2, dim=1)
        x = x * t
        return x

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

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, num_meta_keys, meta_embedding_dims, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = MetaAttention(dim, num_heads, num_meta_keys, meta_embedding_dims)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, metainfo):
        x = x + self.attn(self.norm1(x), metainfo)
        x = x + self.ffn(self.norm2(x))

        return x

from utils.registry import MODEL_REGISTRY
@MODEL_REGISTRY.register()
class MetaRawFormer(nn.Module):
    def __init__(self, 
        in_channels=4, 
        out_channels=4, 
        dim=32,
        layers=4,
        num_meta_keys=4,
        #meta_table_size=[30,26,26,250],
        meta_embedding_dims=384,
        num_blocks=[2,4,4,2], 
        num_refinement_blocks=2,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias'
    ):

        super(MetaRawFormer, self).__init__()
        assert len(num_blocks) == layers and len(heads) == layers
        self.num_meta_keys = num_meta_keys
        #self.meta_embeddings = nn.ModuleList([
        #    nn.Embedding(size, meta_embedding_dims) for size in meta_table_size
        #])
        self.meta_projects = nn.ModuleList([
            nn.Linear(1, meta_embedding_dims) for _ in range(num_meta_keys)
        ])
        
        self.patch_embed = OverlapPatchEmbed(in_channels, dim)

        self.encoders = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(int(dim*2**i), heads[i], num_meta_keys, meta_embedding_dims, ffn_expansion_factor, bias, LayerNorm_type) 
                for _ in range(num_blocks[i])
            ])
            for i in range (layers-1)
        ])
        
        #self.enc_meta_inject = nn.ModuleList([
        #    MetaInjection(in_channels=dim*2**i, num_meta_keys=num_meta_keys, meta_embedding_dims=meta_embedding_dims, num_heads=2**i)
        #    for i in range (layers-1)
        #])
        
        self.middle_block = nn.ModuleList([
            TransformerBlock(int(dim*2**(layers-1)), heads[layers-1], num_meta_keys, meta_embedding_dims, ffn_expansion_factor, bias, LayerNorm_type) 
            for _ in range(num_blocks[layers-1])
        ])
        
        self.decoders = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(int(dim*2**i), heads[i], num_meta_keys, meta_embedding_dims, ffn_expansion_factor, bias, LayerNorm_type) 
                for _ in range(num_blocks[i])
            ])
            for i in range (layers-2, -1, -1)
        ])
        
        #self.dec_meta_inject = nn.ModuleList([
        #    MetaInjection(in_channels=dim*2**i, num_meta_keys=num_meta_keys, meta_embedding_dims=meta_embedding_dims, num_heads=2**i)
        #    for i in range (layers-2, -1, -1)
        #])
        
        self.last_decoder = nn.ModuleList([
            TransformerBlock(int(dim*2**1), heads[0], num_meta_keys, meta_embedding_dims, ffn_expansion_factor, bias, LayerNorm_type) 
            for _ in range(num_blocks[0])
        ])
        
        #self.last_meta_inject = MetaInjection(in_channels=dim*2**1, num_meta_keys=num_meta_keys, meta_embedding_dims=meta_embedding_dims, num_heads=2)

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
            #MetaAwareFuse(in_channels=dim*2**i, num_meta_keys=num_meta_keys, meta_embedding_dims=meta_embedding_dims)
            for i in range (layers-2, -1, -1)
        ])
        self.refinement = nn.ModuleList([
            TransformerBlock(dim*2**1, heads[0], num_meta_keys, meta_embedding_dims, ffn_expansion_factor, bias, LayerNorm_type) 
            for i in range(num_refinement_blocks)
        ])
        self.output = nn.Sequential(
            nn.Conv2d(2*dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, out_channels, 1)
        )

    def forward(self, x, metainfo):
        x = self._check_and_padding(x)

        t = []
        #for i in range(self.num_meta_keys):
        #    embedding = self.meta_embeddings[i](metainfoidx[:, i])
        #    metainfo.append(embedding)
        #metainfo = torch.stack(metainfo,dim=1) #(b,n,d)
        for i in range(self.num_meta_keys):
            t.append(self.meta_projects[i](metainfo[:, i]))

        metainfo = torch.stack(t, dim=1) #(b,n,d)
        shortcut = x
        x = self.patch_embed(x)#c

        encode_features = []
        for encodes, down in zip(self.encoders, self.downs):
            for encode in encodes:
                x = encode(x, metainfo)
            #x = meta_inject(x, metainfo)
            encode_features.append(x)
            x = down(x)
        
        for block in self.middle_block:
            x = block(x, metainfo)
        
        encode_features.reverse()
        for up, fuse, feature, decodes in zip(self.ups[:-1], self.fuses, encode_features[:-1], self.decoders):
            x = up(x)
            x = fuse(x, feature)
            #x = fuse(x, feature, metainfo)
            for decode in decodes:
                x = decode(x, metainfo)
            #x = meta_inject(x, metainfo)
        x = self.ups[-1](x)
        x = torch.cat([x, encode_features[-1]],dim=1)
        for last_block in self.last_decoder:
            x = last_block(x, metainfo)
        #x = self.last_meta_inject(x, metainfo)

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
                        num_meta_keys=4, meta_embedding_dims=384,
                        num_blocks=[2, 2, 2, 4], heads=[1, 2, 4, 8]).cuda()
    #idx_list = [1,2,3,4,2]
    #metainfoidx = torch.tensor(idx_list, dtype=torch.long).unsqueeze(0).cuda()
    metainfo = torch.rand((1,4,1)).cuda()
    x = torch.rand(1,4,512,512).cuda()
    cal_model_complexity(model, x, metainfo)
    exit(0)
    import time
    begin = time.time()
    x = model(x)
    end = time.time()
    print(f'Time comsumed: {end-begin} s')

'''
FLOPs: 170.409429472 G
Params: 9.024678 M
'''