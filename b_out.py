## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange



##########################################################################
## Layer Norm

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

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps
    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class SimpleFuse(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Conv2d(2*in_channels, in_channels, 1)
    def forward(self, x, y):
        return self.body(torch.cat([x,y],dim=1))

##########################################################################
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
        #self.meta_proj = nn.Linear(num_meta_keys*meta_embedding_dims, dim)
        #self.meta_fuse_convs = nn.Sequential(
        #    nn.Conv2d(2*dim, dim, 5, padding=2, groups=dim),
        #    nn.GELU(),
        #    nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        #)
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        '''
        metainfo : (b,n,d)
        '''
        b,c,h,w = x.shape
        
        #metainfo = self.meta_proj(metainfo.view(-1, n*d)).unsqueeze(-1).unsqueeze(-1).expand(b,c,h,w) #(b,c,1,1)
        #x = self.meta_fuse_convs(torch.cat([x, metainfo], dim=1)) #(b,c,h,w)
        
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

'''
The following method is OK for NAFNet baseline
class MetaInjection(nn.Module):
    def __init__(self, in_channels, num_meta_keys, meta_embedding_dims):
        super(MetaInjection, self).__init__()
        self.fc    = nn.Linear(num_meta_keys*meta_embedding_dims, in_channels)
        self.block = NAFBlock(in_channels)
        self.beta  = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)

    def forward(self, x, metainfo):
        #metainfo (b,n,d)
        metainfo = metainfo.flatten(1)#(b, nd)
        gating_factors = torch.sigmoid(self.fc(metainfo))
        gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1)

        f = x * self.gamma + self.beta  # 1) learned feature scaling/modulation
        f = f * gating_factors          # 2) (soft) feature routing based on text
        f = self.block(f)               # 3) block feature enhancement
        return f + x 
'''

class MetaInjection(nn.Module):
    def __init__(self, in_channels, num_meta_keys, dim):
        super(MetaInjection, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(num_meta_keys * dim, num_meta_keys * dim * 2),
            nn.LeakyReLU(),
            nn.Linear(num_meta_keys * dim * 2, in_channels * 2)
        )

    def forward(self, x, metainfo):
        '''
        x (b,c,h,w)
        metainfo (b,n,d)
        '''
        b,n,d = metainfo.shape
        metainfo = metainfo.view(b,n*d) #(b,nd)
        metainfo = self.MLP(metainfo)#(b,2c)
        gamma, beta = metainfo.view(b, -1, 1, 1).chunk(2, dim=1) #(b,c,1,1)
        x = (1 + gamma) * x + beta
        return x
##########################################################################

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

from utils.registry import MODEL_REGISTRY
@MODEL_REGISTRY.register()
class MetaRawFormer(nn.Module):
    def __init__(self, 
        in_channels=4, 
        out_channels=4, 
        dim=32,
        num_meta_keys=5,
        meta_table_size=[30,26,26,250,3],
        meta_embedding_dims=384,
        num_blocks=[2,4,4,2], 
        num_refinement_blocks=2,
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
            nn.Sequential(
                *[TransformerBlock(int(dim*2**i), heads[i], ffn_expansion_factor, bias, LayerNorm_type) 
                for _ in range(num_blocks[i])]
            )
            for i in range (3)
        ])
        
        self.encoder_meta_injects = nn.ModuleList([
            MetaInjection(in_channels=dim*2**i, num_meta_keys=num_meta_keys, dim=meta_embedding_dims)
            for i in range (3)
        ])
        
        self.middle_block = nn.Sequential(
            *[TransformerBlock(int(dim*2**3), heads[3], ffn_expansion_factor, bias, LayerNorm_type) 
            for _ in range(num_blocks[3])]
        )
        
        self.decode_meta_injects = nn.ModuleList([
            MetaInjection(in_channels=dim*2**i, num_meta_keys=num_meta_keys, dim=meta_embedding_dims)
            for i in range (2, 0, -1)
        ])
        
        self.decoders = nn.ModuleList([
            nn.Sequential(
                *[TransformerBlock(int(dim*2**i), heads[i], ffn_expansion_factor, bias, LayerNorm_type) 
                for _ in range(num_blocks[i])]
            )
            for i in range (2, 0, -1)
        ])
        
        self.last_decoder = nn.Sequential(
            *[TransformerBlock(int(dim*2**1), heads[0], ffn_expansion_factor, bias, LayerNorm_type) 
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

        self.refinement = nn.Sequential(
            *[TransformerBlock(dim*2**1, heads[0], ffn_expansion_factor, bias, LayerNorm_type) 
            for i in range(num_refinement_blocks)]
        )
        
        self.last_meta_inject = MetaInjection(in_channels=dim*2, num_meta_keys=num_meta_keys, dim=meta_embedding_dims)
        
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
        for encode, meta_inject, down in zip(self.encoders, self.encoder_meta_injects, self.downs):
            x = encode(x)
            x = meta_inject(x, metainfo)
            encode_features.append(x)
            x = down(x)

        x = self.middle_block(x)
        
        encode_features.reverse()
        for up, fuse, feature, decode, meta_inject in zip(self.ups[:-1], self.fuses, encode_features[:-1], self.decoders, self.decode_meta_injects):
            x = up(x)
            x = fuse(x, feature)
            x = decode(x)
            x = meta_inject(x, metainfo)
        
        x = self.ups[-1](x)
        x = torch.cat([x, encode_features[-1]],dim=1)
        x = self.last_decoder(x) #2c
        x = self.last_meta_inject(x, metainfo) #2c
        x = self.refinement(x)
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
                        num_meta_keys=5, meta_table_size=[30,26,26,250,3], meta_embedding_dims=48,
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
FLOPs: 121.587864576 G
Params: 5.666598 M
'''