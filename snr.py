import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        #initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention4(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # print(q.shape)
        q = self.layer_norm(q)
        k = self.layer_norm(k)
        v = self.layer_norm(v)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        print(q.shape, k.shape, v.shape)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # print(attn.shape, '2')
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        # q = self.layer_norm(q)
        return q, attn


class PositionwiseFeedForward4(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        return x

class EncoderLayer3(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer3, self).__init__()
        self.slf_attn = MultiHeadAttention4(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward4(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class Encoder_patch66(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, d_word_vec=516, n_layers=6, n_head=8, d_k=64, d_v=64,
                d_model=576, d_inner=2048, dropout=0.0, n_position=10, scale_emb=False):
        # 2048
        super().__init__()

        self.n_position = n_position
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer3(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.count = 0
        self.center_example = None
        self.center_coordinate = None

    def forward(self, src_fea, src_location, return_attns=False, src_mask=None):
        enc_output = src_fea
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
        return enc_output

###############################
class low_light_transformer(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                predeblur=False, HR_in=False, w_TSA=True):
        super(low_light_transformer, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)

        if self.HR_in:
            self.conv_first_1 = nn.Conv2d(4, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(4, nf, 3, 1, 1, bias=True)

        self.feature_extraction = make_layer(ResidualBlock_noBN_f, front_RBs)
        self.recon_trunk = make_layer(ResidualBlock_noBN_f, back_RBs)

        self.upconv1 = nn.Conv2d(nf*2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf*2, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64*2, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 4, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.transformer = Encoder_patch66(d_model=1024, d_inner=2048, n_layers=6)
        self.recon_trunk_light = make_layer(ResidualBlock_noBN_f, 6)

    def forward(self, x, mask=None):
        x_center = x

        L1_fea_1 = self.lrelu(self.conv_first_1(x_center))#(b,c,h,w)
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))#(b,c,h//2,w//2)
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))#(b,c,h//4,w//4)

        fea = self.feature_extraction(L1_fea_3)#(b,c,h//4,w//4)
        fea_light = self.recon_trunk_light(fea)

        h_feature = fea.shape[2]
        w_feature = fea.shape[3]
        mask = F.interpolate(mask, size=[h_feature, w_feature], mode='nearest')

        xs = np.linspace(-1, 1, fea.size(3) // 4)
        ys = np.linspace(-1, 1, fea.size(2) // 4)
        xs = np.meshgrid(xs, ys)
        xs = np.stack(xs, 2)
        xs = torch.Tensor(xs).unsqueeze(0).repeat(fea.size(0), 1, 1, 1).cuda()
        xs = xs.view(fea.size(0), -1, 2)

        height = fea.shape[2]
        width = fea.shape[3]
        print(mask.shape, fea.shape)
        fea_unfold = F.unfold(fea, kernel_size=4, dilation=1, stride=4, padding=0)#torch.Size([b, 16c, h//16w//16])
        fea_unfold = fea_unfold.permute(0, 2, 1)#torch.Size([b, h//16w//16, 16c])

        mask_unfold = F.unfold(mask, kernel_size=4, dilation=1, stride=4, padding=0)
        mask_unfold = mask_unfold.permute(0, 2, 1)
        mask_unfold = torch.mean(mask_unfold, dim=2).unsqueeze(dim=-2)
        mask_unfold[mask_unfold <= 0.5] = 0.0
        print(fea_unfold.shape)
        fea_unfold = self.transformer(fea_unfold, xs, src_mask=mask_unfold)#(b, h//16w//16, 16c)
        fea_unfold = fea_unfold.permute(0, 2, 1)#(b, 16c, h//16w//16)
        fea_unfold = nn.Fold(output_size=(height, width), kernel_size=(4, 4), stride=4, padding=0, dilation=1)(fea_unfold)#(b,c,h//4,w//4)
        
        channel = fea.shape[1]
        mask = mask.repeat(1, channel, 1, 1)
        fea = fea_unfold * (1 - mask) + fea_light * mask #(b,c,h//4,w//4)

        out_noise = self.recon_trunk(fea) #(b,c,h//4,w//4)
        out_noise = torch.cat([out_noise, L1_fea_3], dim=1)#(b,2c,h//4,w//4)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))#(b,c,h//2,w//2)
        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)#(b,2c,h//2,w//2)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))#(b,c,h,w)
        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)#(b,2c,h,w)
        out_noise = self.lrelu(self.HRconv(out_noise))#(b,c,h,w)
        out_noise = self.conv_last(out_noise)#(b,c_in,h,w)
        out_noise = out_noise + x_center

        return out_noise

def cal_model_complexity(model, x, mask):
    import thop
    flops, params = thop.profile(model, inputs=(x,mask,), verbose=False)
    print(f"FLOPs: {flops / 1e9} G")
    print(f"Params: {params / 1e6} M")

if __name__ == '__main__':
    model = low_light_transformer(nf=64,nframes=5,groups=8,front_RBs=1,
                                    back_RBs=1, predeblur=True, HR_in=True, w_TSA=True).cuda()
    #x = torch.rand(1,3,512,512).cuda()
    mask = torch.rand(1,1,512,512).cuda()
    x = torch.rand(1,4,1024,1024).cuda()
    with torch.no_grad():
        cal_model_complexity(model, x, mask)
        exit(0)
        import time
        begin = time.time()
        x = model(x, mask)
        print(x.shape)
        end = time.time()
        print(f'Time comsumed: {end-begin} s')