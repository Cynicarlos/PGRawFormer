import math
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


#======================NAFBlock==================================
class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out

def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)

class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)

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

#======================NAFBlock==================================

class SimpleFuse(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Conv2d(2*in_channels, in_channels, 1)
    def forward(self, x, y):
        return self.body(torch.cat([x,y],dim=1))

'''
fusion v1
class MetaAwareFuse(nn.Module):
    def __init__(self, in_channels, num_meta_keys):
        super().__init__()
        self.fc = nn.Linear(num_meta_keys, in_channels)
        self.alpha = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, 1),
            nn.GELU()
        )
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels, 2*in_channels, 3, padding=1, groups=in_channels)
        self.conv3 = nn.Conv2d(in_channels, 2*in_channels, 3, padding=1, groups=in_channels)
        self.conv4 = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.GELU()
        )

    def forward(self, x, m):
        #x, y (b,c,h,w)
        #m (b,n,1)
        b,c,h,w = x.shape
        m = m.flatten(1)#(b, n)
        m = self.fc(m)#(b, c)
        m = m.unsqueeze(-1).unsqueeze(-1).expand(b,c,h,w) #(b, c, h, w)
        t = self.conv1(torch.cat([x,m],dim=1)) #c
        m = self.conv2(m)#2c
        m1, m2 = m.chunk(2, dim=1) #c,c
        x = self.conv3(x)#2c
        x1, x2 = x.chunk(2, dim=1) #c,c
        x = torch.cat([self.act1(m1)*x2, m2*self.act2(x1)],dim=1)#2c
        x = self.conv4(x)#c
        x = self.alpha * x + self.beta * t
        return x
'''

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

class MetaAttention(nn.Module):
    def __init__(self, dim, num_heads, meta_dims):
        super(MetaAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Linear(meta_dims, dim)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x, metainfo):
        #x (b,c,h,w)
        #metainfo (b,d)
        b,c,h,w = x.shape
        _, d = metainfo.shape
        metainfo = self.q(metainfo) #(b,c)
        metainfo = metainfo.unsqueeze(-1).unsqueeze(-1).expand(b,c,h,w)
        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1) #(b,c,h,w)
        q = rearrange(metainfo, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)#(b,head,c,hw)
        k = torch.nn.functional.normalize(k, dim=-1)#(b,head,c,hw)
        attn = (q @ k.transpose(-2, -1)) * self.temperature#(b,head,c,c)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)#(b,c,h,w)
        out = self.project_out(out)
        return out


#seems better than fusion v1
class MetaAwareFuse(nn.Module):
    def __init__(self, in_channels, meta_dims):
        super().__init__()
        self.fc = nn.Linear(meta_dims, in_channels)
        self.alpha = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 2*in_channels, 3, padding=1, groups=in_channels),
            nn.GELU(),
            nn.Conv2d(2*in_channels, in_channels, 1),
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(2*in_channels, 2*in_channels, 1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 1)
        self.ca = ChannelAttention(2*in_channels)
        self.conv4 = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1),
        )

    def forward(self, x, m):
        #x, y (b,c,h,w)
        #m (b,d)
        b,c,h,w = x.shape
        m = self.fc(m)#(b, c)
        m = m.unsqueeze(-1).unsqueeze(-1).expand(b,c,h,w) #(b, c, h, w)
        t = self.alpha * m * x
        xt = self.conv1(x)#c
        x = torch.cat([x, m],dim=1)#2c
        x = self.conv2(x)#2c
        x1, x2 = x.chunk(2, dim=1)#c,c
        x = self.act(x1) * self.conv3(x2) + xt
        x = torch.cat([x, t],dim=1)#2c
        x = self.conv4(self.ca(x))
        return x


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

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(EncoderBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

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

class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(DecoderBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        #self.attn = MetaAttention(dim, num_heads, meta_dims)
        #self.attn = MultiheadDiffAttn(embed_dim=dim, depth=depth, num_heads=num_heads)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        #self.ffn = MetaAwareFuse(dim, meta_dims)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, metainfo):
        x = x + self.attn(self.norm1(x))
        #x = x + self.attn(self.norm1(x), metainfo)
        x = x + self.ffn(self.norm2(x), metainfo)
        #x = x + self.ffn(self.norm2(x), metainfo)
        return x

class AEC(nn.Module):
    def __init__(self, in_channels, meta_dims):
        super().__init__()
        self.fc = nn.Linear(meta_dims, in_channels)
        self.alpha = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 2*in_channels, 3, padding=1, groups=in_channels),
            nn.GELU(),
            nn.Conv2d(2*in_channels, in_channels, 1),
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(2*in_channels, 2*in_channels, 1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 1)
        self.ca = ChannelAttention(2*in_channels)
        self.conv4 = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1),
        )

    def forward(self, x, m):
        #x, y (b,c,h,w)
        #m (b,d)
        b,c,h,w = x.shape
        m = self.fc(m)#(b, c)
        m = m.unsqueeze(-1).unsqueeze(-1).expand(b,c,h,w) #(b, c, h, w)
        t = self.alpha * m * x
        xt = self.conv1(x)#c
        x = torch.cat([x, m],dim=1)#2c
        x = self.conv2(x)#2c
        x1, x2 = x.chunk(2, dim=1)#c,c
        x = self.act(x1) * self.conv3(x2) + xt
        x = torch.cat([x, t],dim=1)#2c
        x = self.conv4(self.ca(x))
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
        meta_dims=128,
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
        
        self.meta_embedding = nn.Linear(num_meta_keys, meta_dims)
        self.patch_embed = OverlapPatchEmbed(in_channels, dim)

        '''
        self.aec = AEC(in_channels=dim, meta_dims=meta_dims)
        
        self.aec_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(int(dim*2**i), int(dim*2**i), 1),
                nn.GELU()
            )
            for i in range (layers-2, -1, -1)
        ])
        
        self.aec_downs = nn.ModuleList([
            Downsample(n_feat=dim*2**i)
            for i in range (layers-1)
        ])
        
        self.aec_down_fuses = nn.ModuleList([
            SimpleFuse(in_channels=dim*2**i)
            for i in range (layers-1)
        ])
        
        self.aec_mid_fuse = SimpleFuse(in_channels=dim*2**3)
        
        self.aec_up_fuses = nn.ModuleList([
            SimpleFuse(in_channels=dim*2**i)
            for i in range (layers-2, -1, -1)
        ])
        '''
        
        self.encoders = nn.ModuleList([
            nn.ModuleList([
                #DecoderBlock(int(dim*2**i), heads[i], meta_dims, ffn_expansion_factor, bias, LayerNorm_type)
                TransformerBlcok(int(dim*2**i), heads[i], ffn_expansion_factor, bias, LayerNorm_type)
                for _ in range(num_blocks[i])
            ])
            for i in range (layers-1)
        ])
        
        #self.encoder_meta_fuses = nn.ModuleList([
        #    MetaAwareFuse(int(dim*2**i), meta_dims)
        #    for i in range (layers-1)
        #])
        
        self.middle_block = nn.ModuleList([
            #DecoderBlock(int(dim*2**(layers-1)), heads[layers-1], meta_dims, ffn_expansion_factor, bias, LayerNorm_type)
            TransformerBlcok(int(dim*2**(layers-1)), heads[layers-1], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[layers-1])
        ])
        
        #self.mid_meta_fuse = MetaAwareFuse(int(dim*2**(layers-1)), meta_dims)

        
        self.decoders = nn.ModuleList([
            nn.ModuleList([
                #DecoderBlock(int(dim*2**i), heads[i], meta_dims, ffn_expansion_factor, bias, LayerNorm_type)
                TransformerBlcok(int(dim*2**i), heads[i], ffn_expansion_factor, bias, LayerNorm_type)
                for _ in range(num_blocks[i])
            ])
            for i in range (layers-2, -1, -1)
        ])
        
        self.decoder_meta_fuses = nn.ModuleList([
            MetaAwareFuse(int(dim*2**i), meta_dims)
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
            #DecoderBlock(2*dim, heads[0], meta_dims, ffn_expansion_factor, bias, LayerNorm_type)
            TransformerBlcok(2*dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
            for i in range(num_refinement_blocks)
        ])
        
        #self.refine_meta_fuses = MetaAwareFuse(2*dim, meta_dims)

        self.output = nn.Sequential(
            nn.Conv2d(2*dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, out_channels, 1)
        )

    def forward(self, x, metainfo):
        #matainfo : (b,n,1)
        x = self._check_and_padding(x)

        #t = []
        #for i in range(self.num_meta_keys):
        #    embedding = self.meta_embeddings[i](metainfoidx[:, i])
        #    metainfo.append(embedding)
        #metainfo = torch.stack(metainfo,dim=1) #(b,n,d)
        #for i in range(self.num_meta_keys):
        #    t.append(self.meta_projects[i](metainfo[:, i]))

        #metainfo = torch.stack(t, dim=1) #(b,n,d)
        _,n,d = metainfo.shape
        metainfo = self.meta_embedding(metainfo.view(-1, n)) #(b, d)
        shortcut = x
        x = self.patch_embed(x)#c

        #aec_feature = self.aec(x, metainfo)
        #aec_features = []

        encode_features = []
        for encodes, down in zip(self.encoders, self.downs):
            for encode in encodes:
                x = encode(x)
            #x = meta_fuse(x, metainfo)
            #x = aec_fuse(x, aec_feature)
            encode_features.append(x)
            #aec_features.append(aec_feature)
            x = down(x)
            #aec_feature = aec_down(aec_feature)

        for block in self.middle_block:
            x = block(x)
            #x = block(x, metainfo)
        #x = self.aec_mid_fuse(x, aec_feature)
        #x = self.mid_meta_fuse(x, metainfo)
        
        encode_features.reverse()
        #aec_features.reverse()
        
        for up, fuse, feature, decodes, meta_fuse in zip(
            self.ups, self.fuses, encode_features, self.decoders, self.decoder_meta_fuses
        ):
            x = up(x)
            x = fuse(x, feature)
            for decode in decodes:
                x = decode(x)
            x = meta_fuse(x, metainfo)
                #x = decode(x, metainfo)
            #x = aec_fuse(x, aec_conv(aec_feature))

        x = self.refine_conv(x)
        for refine_block in self.refinement:
            x = refine_block(x)
            #x = refine_block(x, metainfo)
        
        #x = self.refine_meta_fuses(x, metainfo)
        
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
                        num_meta_keys=4, num_blocks=[2,2,2,2], num_refinement_blocks=2, heads=[1, 2, 4, 8]).cuda()
    #idx_list = [1,2,3,4,2]
    #metainfoidx = torch.tensor(idx_list, dtype=torch.long).unsqueeze(0).cuda()
    metainfo = torch.rand((1,4,1)).cuda()
    x = torch.rand(1,4,1024,1024).cuda()
    with torch.no_grad():
        cal_model_complexity(model, x, metainfo)
        exit(0)
        import time
        begin = time.time()
        x = model(x)
        end = time.time()
        print(f'Time comsumed: {end-begin} s')