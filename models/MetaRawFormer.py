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

T_MAX = 512*512

from torch.utils.cpp_extension import load
wkv_cuda = load(name="wkv", sources=["./models/cuda/wkv_op.cpp", "./models/cuda/wkv_cuda.cu"],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)

def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())

class OmniShift(nn.Module):
    def __init__(self, dim):
        super(OmniShift, self).__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias=False) 
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True) 
        

        # Define the layers for testing
        self.conv5x5_reparam = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias = False) 
        self.repram_flag = True

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x) 
        # import pdb 
        # pdb.set_trace() 
        out = self.alpha[0]*x + self.alpha[1]*out1x1 + self.alpha[2]*out3x3 + self.alpha[3]*out5x5
        return out

    def reparam_5x5(self):
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution 
        
        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2)) 
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1)) 
        
        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2)) 
        
        combined_weight = self.alpha[0]*identity_weight + self.alpha[1]*padded_weight_1x1 + self.alpha[2]*padded_weight_3x3 + self.alpha[3]*self.conv5x5.weight 
        
        device = self.conv5x5_reparam.weight.device 

        combined_weight = combined_weight.to(device)

        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)


    def forward(self, x): 
        if self.training: 
            self.repram_flag = True
            out = self.forward_train(x) 
        elif self.training == False and self.repram_flag == True:
            self.reparam_5x5() 
            self.repram_flag = False 
            out = self.conv5x5_reparam(x)
        elif self.training == False and self.repram_flag == False:
            out = self.conv5x5_reparam(x)
        
        return out 

class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, key_norm=False):
        super().__init__()
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        
        self.dwconv = nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1, groups=n_embd, bias=False) 
        
        self.recurrence = 2 
        
        self.omni_shift = OmniShift(dim=n_embd)

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False) 


        with torch.no_grad():
            self.spatial_decay = nn.Parameter(torch.randn((self.recurrence, self.n_embd))) 
            self.spatial_first = nn.Parameter(torch.randn((self.recurrence, self.n_embd))) 

    def jit_func(self, x, resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        h, w = resolution

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')    


        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, resolution):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, resolution) 
        
        for j in range(self.recurrence): 
            if j%2==0:
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v) 
            else:
                h, w = resolution 
                k = rearrange(k, 'b (h w) c -> b (w h) c', h=h, w=w) 
                v = rearrange(v, 'b (h w) c -> b (w h) c', h=h, w=w) 
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v) 
                k = rearrange(k, 'b (w h) c -> b (h w) c', h=h, w=w) 
                v = rearrange(v, 'b (w h) c -> b (h w) c', h=h, w=w) 
                

        x = v
        if self.key_norm is not None:
            x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x

class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, hidden_rate=4,key_norm=False):
        super().__init__()
        self.n_embd = n_embd
        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False) 
        
        self.omni_shift = OmniShift(dim=n_embd)
        
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x, resolution):

        h, w = resolution

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')    

        k = self.key(x)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv 

        return x

class Block(nn.Module):
    def __init__(self, n_embd, hidden_rate=4, key_norm=False):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd) 

        self.att = VRWKV_SpatialMix(n_embd, key_norm=key_norm)

        self.ffn = VRWKV_ChannelMix(n_embd, hidden_rate, key_norm=key_norm)

        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)

    def forward(self, x): 
        b, c, h, w = x.shape
        
        resolution = (h, w)

        # x = self.dwconv1(x) + x
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.gamma1 * self.att(self.ln1(x), resolution) 
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        # x = self.dwconv2(x) + x
        x = rearrange(x, 'b c h w -> b (h w) c')    
        x = x + self.gamma2 * self.ffn(self.ln2(x), resolution) 
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x

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

class MetaInjection(nn.Module):
    def __init__(self, in_channels, num_meta_keys, meta_embedding_dims):
        super(MetaInjection, self).__init__()
        self.fc    = nn.Linear(num_meta_keys*meta_embedding_dims, in_channels)
        self.beta  = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.ffn = FeedForward(dim=in_channels, ffn_expansion_factor=2, bias=True)
    def forward(self, x, metainfo):
        metainfo = metainfo.flatten(1)#(b, nd)
        gating_factors = torch.sigmoid(self.fc(metainfo))
        gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1)
        f = x * self.gamma + self.beta  # 1) learned feature scaling/modulation
        f = f * gating_factors          # 2) (soft) feature routing based on text
        f = self.ffn(f)               # 3) block feature enhancement
        return f + x

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

class MetaTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, num_meta_keys, meta_embedding_dims, ffn_expansion_factor, bias, LayerNorm_type):
        super(MetaTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.meta_attn = MetaAttention(dim, num_heads, num_meta_keys, meta_embedding_dims)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, metainfo):
        x = x + self.meta_attn(self.norm1(x), metainfo)
        x = x + self.ffn(self.norm2(x))

        return x

from utils.registry import MODEL_REGISTRY
@MODEL_REGISTRY.register()
class MetaRawFormer(nn.Module):
    def __init__(self, 
        in_channels=4, 
        out_channels=4, 
        dim=48,
        layers=4,
        num_meta_keys=4,
        #meta_table_size=[30,26,26,250],
        meta_embedding_dims=384,
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
        #self.meta_embeddings = nn.ModuleList([
        #    nn.Embedding(size, meta_embedding_dims) for size in meta_table_size
        #])
        self.meta_projects = nn.ModuleList([
            nn.Linear(1, meta_embedding_dims) for _ in range(num_meta_keys)
        ])
        
        self.patch_embed = OverlapPatchEmbed(in_channels, dim)

        self.encoders = nn.ModuleList([
            nn.ModuleList([
                #TransformerBlock(int(dim*2**i), heads[i], ffn_expansion_factor, bias, LayerNorm_type) 
                Block(n_embd=int(dim*2**i))
                for _ in range(num_blocks[i])
            ])
            for i in range (layers-1)
        ])
        
        #self.enc_meta_inject = nn.ModuleList([
        #    MetaInjection(in_channels=dim*2**i, num_meta_keys=num_meta_keys, meta_embedding_dims=meta_embedding_dims, num_heads=2**i)
        #    for i in range (layers-1)
        #])
        
        self.middle_block = nn.ModuleList([
            #TransformerBlock(int(dim*2**(layers-1)), heads[layers-1], ffn_expansion_factor, bias, LayerNorm_type) 
            Block(n_embd=int(dim*2**(layers-1)))
            for _ in range(num_blocks[layers-1])
        ])
        
        self.decoders = nn.ModuleList([
            nn.ModuleList([
                Block(n_embd=int(dim*2**i))
                #TransformerBlock(int(dim*2**i), heads[i], ffn_expansion_factor, bias, LayerNorm_type) 
                #MetaTransformerBlock(int(dim*2**i), heads[i], num_meta_keys, meta_embedding_dims, ffn_expansion_factor, bias, LayerNorm_type) 
                for _ in range(num_blocks[i])
            ])
            for i in range (layers-2, -1, -1)
        ])
        
        self.dec_meta_inject = nn.ModuleList([
            MetaInjection(in_channels=dim*2**i, num_meta_keys=num_meta_keys, meta_embedding_dims=meta_embedding_dims)
            for i in range (layers-2, -1, -1)
        ])
        
        self.last_decoder = nn.ModuleList([
            Block(n_embd=int(dim*2**1))
            #TransformerBlock(int(dim*2**1), heads[0], ffn_expansion_factor, bias, LayerNorm_type) 
            #MetaTransformerBlock(int(dim*2**1), heads[0], num_meta_keys, meta_embedding_dims, ffn_expansion_factor, bias, LayerNorm_type) 
            for _ in range(num_blocks[0])
        ])

        self.last_meta_inject = MetaInjection(in_channels=dim*2**1, num_meta_keys=num_meta_keys, meta_embedding_dims=meta_embedding_dims)

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
            Block(n_embd=int(dim*2**1))
            #TransformerBlock(dim*2**1, heads[0], ffn_expansion_factor, bias, LayerNorm_type) 
            #MetaTransformerBlock(dim*2**1, heads[0], num_meta_keys, meta_embedding_dims, ffn_expansion_factor, bias, LayerNorm_type) 
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
                x = encode(x)
            #x = meta_inject(x, metainfo)
            encode_features.append(x)
            x = down(x)
        
        for block in self.middle_block:
            x = block(x)
        
        encode_features.reverse()
        for up, fuse, feature, decodes, meta_inject in zip(self.ups[:-1], self.fuses, encode_features[:-1], self.decoders, self.dec_meta_inject):
            x = up(x)
            x = fuse(x, feature)
            #x = fuse(x, feature, metainfo)
            for decode in decodes:
                x = decode(x)
            x = meta_inject(x, metainfo)
            #x = meta_inject(x, metainfo)
        x = self.ups[-1](x)
        x = torch.cat([x, encode_features[-1]],dim=1)
        for last_block in self.last_decoder:
            x = last_block(x)
        #x = self.last_meta_inject(x, metainfo)
        x = self.last_meta_inject(x, metainfo)

        for refine_block in self.refinement:
            x = refine_block(x)
        
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
    model = MetaRawFormer(in_channels=4, out_channels=4, dim=48, layers=4,
                        num_meta_keys=4, meta_embedding_dims=384,
                        num_blocks=[4,6,6,8], num_refinement_blocks=4, heads=[1, 2, 4, 8]).cuda()
    #idx_list = [1,2,3,4,2]
    #metainfoidx = torch.tensor(idx_list, dtype=torch.long).unsqueeze(0).cuda()
    metainfo = torch.rand((1,4,1)).cuda()
    x = torch.rand(1,4,256,256).cuda()
    cal_model_complexity(model, x, metainfo)
    exit(0)
    import time
    begin = time.time()
    x = model(x)
    end = time.time()
    print(f'Time comsumed: {end-begin} s')