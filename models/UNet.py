import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/root/autodl-tmp/Generalization')

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        # 平均池化和最大池化，用于获取全局信息
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 7x7卷积捕捉空间信息
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        
        # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 获取输入的维度信息
        b, c, h, w = x.size()
        
        # 沿通道维度获取全局平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (b, 1, h, w)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (b, 1, h, w)
        
        # 将平均池化和最大池化结果进行拼接
        pool_out = torch.cat([avg_out, max_out], dim=1)  # (b, 2, h, w)
        
        # 使用7x7卷积提取空间特征
        spatial_attn = self.spatial_conv(pool_out)  # (b, 1, h, w)
        
        # 使用Sigmoid激活函数生成空间注意力权重
        attn_weights = self.sigmoid(spatial_attn)  # (b, 1, h, w)
        
        # 将空间注意力权重扩展为与输入相同的形状
        attn_weights = attn_weights.expand(b, c, h, w)  # (b, c, h, w)
        
        # 将输入特征与空间注意力权重相乘
        output = x * attn_weights  # (b, c, h, w)
        
        return output

class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class SimpleFuse(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GELU()
        )
        self.convs = nn.Sequential(
            nn.Conv2d(2*in_channels, 2*in_channels, 3, padding=1, groups=2*in_channels),
            ChannelAttention(num_feat=2*in_channels),
            nn.Conv2d(2*in_channels, in_channels, 1)     
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GELU()
        )
        self.out = nn.Conv2d(in_channels, in_channels, 1)
    def forward(self, cur, pre):
        x = torch.cat([cur, pre], dim=1)
        pre = self.conv1(pre) #c
        x = self.convs(x) * pre
        x = self.conv2(x) + cur
        x = self.out(x)
        return x

class MIF(nn.Module):
    def __init__(self, feature_dim, embedding_dim=32, num_heads=8):
        super(MIF, self).__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.feature_dim = feature_dim
        
        self.ln0 = LayerNorm(feature_dim, data_format="channels_first")
        self.ln1 = LayerNorm(feature_dim, data_format="channels_first")
        self.query = nn.Linear(embedding_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        # 为每个头自适应地学习一个权重
        self.head_weights = nn.Parameter(torch.ones(num_heads))
        
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x, embeddings):
        b, c, h, w = x.shape
        x  = self.ln0(x) #(b,c,h,w)
        x_flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c)

        Q = self.query(embeddings).view(b, 5, self.num_heads, self.head_dim)  # (b, 5, num_heads, head_dim)
        K = self.key(x_flat).view(b, h * w, self.num_heads, self.head_dim)  # (b, h*w, num_heads, head_dim)
        V = self.value(x_flat).view(b, h * w, self.num_heads, self.head_dim)  # (b, h*w, num_heads, head_dim)

        Q = Q.permute(0, 2, 1, 3)  # (b, num_heads, 5, head_dim)
        K = K.permute(0, 2, 3, 1)  # (b, num_heads, head_dim, h*w)
        V = V.permute(0, 2, 1, 3)  # (b, num_heads, h*w, head_dim)

        # 计算注意力权重
        attn_weights = torch.matmul(Q, K) / (self.head_dim ** 0.5)  # (b, num_heads, 5, h*w)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # 自适应调整每个头的权重
        adaptive_weights = torch.softmax(self.head_weights, dim=0).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        attn_output = torch.matmul(attn_weights, V) * adaptive_weights  # (b, num_heads, 5, head_dim)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(b, 5, self.feature_dim)  # (b, 5, feature_dim)

        attn_output = self.out_proj(attn_output)  # (b, 5, feature_dim)

        attn_output = attn_output.permute(0, 2, 1).unsqueeze(3).unsqueeze(4)  # (b, feature_dim, 5, 1, 1)
        attn_output = attn_output.expand(-1, -1, -1, h, w)  # (b, feature_dim, 5, h, w)

        attn_output = attn_output.mean(dim=2)  # (b, feature_dim, h, w)
        attn_output = self.ln1(attn_output)
        return x + attn_output  # 融合输入与注意力输出

class DNTree(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.left_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GELU()
        )
        self.right_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GELU()
        )
        self.left_ca = ChannelAttention(num_feat=in_channels)
        self.left_sa = SpatialAttention(kernel_size=7)
        
        self.right_ca = ChannelAttention(num_feat=in_channels)
        self.right_sa = SpatialAttention(kernel_size=7)
        
        self.in_branch_conv1 = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GELU(),
        )
        self.in_branch_conv2 = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GELU(),
        )
        
        self.cross_branch_conv1 = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GELU(),
        )
        self.cross_branch_conv2 = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GELU(),
        )
        
        self.ca = ChannelAttention(2*in_channels)
        self.conv_out = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, 3, padding=1),
            nn.GELU()
        )
        
    def forward(self, x):
        shortcut = x
        l = self.left_conv(x)
        r = self.right_conv(x)
        in1 = torch.cat([self.left_ca(l), self.left_sa(l)], dim=1)
        in2 = torch.cat([self.right_ca(r), self.right_sa(r)], dim=1)
        cross1 = torch.cat([self.left_ca(l), self.right_sa(r)], dim=1)
        cross2 = torch.cat([self.left_sa(l), self.right_ca(r)], dim=1)
        l = self.in_branch_conv1(in1) + self.in_branch_conv2(in2)
        r = self.cross_branch_conv1(cross1) + self.cross_branch_conv2(cross2)
        x = torch.cat([l,r],dim=1)
        x = self.ca(x)
        x = self.conv_out(x)
        x = x + shortcut
        return x

from utils.registry import MODEL_REGISTRY
@MODEL_REGISTRY.register()
class UNet(nn.Module):
    def __init__(self, in_channels, base_channels, embedding_dim=32):
        super(UNet, self).__init__()
        self.ExposureTime_dict=[
            None, '1/3200', '1/2000', '1/1600', '1/1000', '1/800', '1/500', '1/400', '1/250', 
            '1/200', '1/160', '1/100', '1/80', '1/50', '1/40','1/30','1/25', '1/20', '1/15', 
            '1/10', '1/8', '1/5', '1/4', '2/5', '1/2', '2', '16/5', '4',  '10', '30'
        ]
        self.Fnumber_dict = [None, '16/5', '4', '9/2', '5', '28/5', '63/10', '71/10', '8', '9', '10', '11', '13', '14', '16', '18', '22']
        self.FocalLength_dict = [
            None, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 
            39, 41, 42, 44,  45, 46, 47, 48, 49, 50, 53, 54, 57, 59, 61, 63,
            66, 67, 73, 78, 82, 83, 87, 89, 90, 120, 128, 139, 166, 181, 223]
        self.ISOSpeedRating_dict = [None, 50, 64, 80, 100, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 
                              1600, 2000, 2500, 3200, 4000, 5000, 6400, 8000, 10000, 12800, 16000, 25600]
        self.MeteringMode_dict = [None, 'CenterWeightedAverage', 'Pattern']
        self.dicts = {
            'ExposureTime': self.ExposureTime_dict,
            'Fnumber': self.Fnumber_dict,
            'FocalLength': self.FocalLength_dict,
            'ISOSpeedRating': self.ISOSpeedRating_dict,
            'MeteringMode': self.MeteringMode_dict
        }
        self.dicts_keys = list(self.dicts.keys()) #['ExposureTime','Fnumber','FocalLength','ISOSpeedRating','MeteringMode']
        self.embedding_layers = nn.ModuleDict({
            key: nn.Embedding(len(dict_), embedding_dim)
            for key, dict_ in self.dicts.items()
        })
        self.encode_mifs = nn.ModuleList([
            MIF(feature_dim=base_channels*(2**i), embedding_dim=embedding_dim, num_heads=2**i)
            for i in range(4)
        ])
        
        self.decode_mifs = nn.ModuleList([
            MIF(feature_dim=base_channels*(2**i), embedding_dim=embedding_dim, num_heads=2**i)
            for i in range(2, -1, -1)
        ])

        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GELU()
        )
        
        self.encode_convs = nn.ModuleList([
            DNTree(base_channels*(2**i))
            for i in range(4)
        ])
        self.downs = nn.ModuleList([
            nn.Conv2d(base_channels*(2**i), base_channels*(2**(i+1)), kernel_size=2, stride=2)
            for i in range(3)
        ])
        self.decode_convs = nn.ModuleList([
            DNTree(base_channels*(2**i))
            for i in range(2, -1, -1)
        ])
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(base_channels*(2**(i+1)), base_channels*(2**i), kernel_size=2, stride=2)
            for i in range(2, -1, -1)
        ])
        self.simplefuses = nn.ModuleList([
            SimpleFuse(in_channels=base_channels*(2**i))
            for i in range(2, -1, -1)
        ])
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x, metaInfoIdx):
        x = self._check_and_padding(x)
        embeddings = []
        for i, key in enumerate(self.dicts_keys):
            embedding = self.embedding_layers[key](metaInfoIdx[:, i])
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings, dim=1)
        
        x = self.conv_in(x) #c
        encoder_features = []
        for mif, encode, down in zip(self.encode_mifs[:-1], self.encode_convs[:-1], self.downs):
            x = mif(x, embeddings)
            x = encode(x)
            encoder_features.append(x)
            x = down(x)
        x = self.encode_mifs[-1](x, embeddings)
        x = self.encode_convs[-1](x)
        
        encoder_features.reverse()
        for up, fuse, feature, mif, decode in zip(self.ups, self.simplefuses, encoder_features, self.decode_mifs, self.decode_convs):
            x = up(x)
            x = fuse(x, feature)
            x = mif(x, embeddings)
            x = decode(x)
        x = self.conv_out(x)
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

'''
FLOPs: 85.8782464 G
Params: 4.708476 M
'''

def cal_model_complexity(model, x, metainfoidx):
    import thop
    flops, params = thop.profile(model, inputs=(x, metainfoidx,), verbose=False)
    print(f"FLOPs: {flops / 1e9} G")
    print(f"Params: {params / 1e6} M")

if __name__ == '__main__':
    idx_list = [1,2,3,4,2]
    metainfoidx = torch.tensor(idx_list, dtype=torch.long).view(-1, 1).unsqueeze(0).cuda()
    model = UNet(in_channels=4, base_channels=32, embedding_dim=128).cuda()
    #x = torch.rand(1,4,512,512).cuda()
    x = torch.rand(1,4,512,512).cuda()
    cal_model_complexity(model, x, metainfoidx)
    import time
    begin = time.time()
    x = model(x, metainfoidx)
    end = time.time()
    print(f'Time comsumed: {end-begin} s')