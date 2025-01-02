import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
x = torch.tensor([
    [
        [
            [1,1,2,2,3,3],
            [2,2,2,2,3,3],
            [4,4,5,5,6,6],
            [4,4,5,5,6,6],
            [7,7,8,8,9,9],
            [7,7,8,8,9,9]
        ],
        [
            [1,1,2,2,3,3],
            [1,1,2,2,3,3],
            [4,4,5,5,6,6],
            [4,4,5,5,6,6],
            [7,7,8,8,9,9],
            [7,7,8,8,7,7]
        ]
    ]
], dtype=torch.float32)
b,c,h,w = x.shape
ps = (2,2)
num_h_patch, num_w_patch = math.ceil(h/ps[0]), math.ceil(w/ps[1])
new_h, new_w = num_h_patch*ps[0], num_w_patch*ps[1]
x = F.interpolate(x, (new_h, new_w))#(1,1,6,6)

x_unfolded = F.unfold(x, kernel_size=ps, stride=ps[0])#(b, c*ps[0]*ps[1], num_patch) (1,8,9)
x_unfolded = rearrange(x_unfolded, 'b (c l) n -> b c l n', c=c, l=ps[0]*ps[1])#(b, c, ps[0]*ps[1], num_patch) (1,2,4,9)
x_unfolded = rearrange(x_unfolded, 'b c (h w) n -> b c n h w', h=ps[0], w=ps[1])#(b, c, num_patch, ps, ps) (1,2,9,2,2)


x_unfolded = torch.mean(x_unfolded, dim=(-2,-1))#(b, c, num_patch) (1,2,9)

x_unfolded = rearrange(x_unfolded, 'b c (h w) -> b c h w', h=num_h_patch, w=num_w_patch)#(b,c,num_h_patch,num_w_patch) (1,2,3,3)

#x_unfolded = SA(x_unfolded) (b,c,num_h_patch,num_w_patch) (1,2,3,3)

x_unfolded = rearrange(x_unfolded, 'b c h w -> b c (h w)') #(b,c,num_h_patch,num_w_patch) (1,2,9)
x_unfolded = x_unfolded.unsqueeze(-1).unsqueeze(-1)#(b,c,num_h_patch,num_w_patch,1,1) (1,2,9,1,1)
x_unfolded = x_unfolded.repeat(1, 1, 1, 2, 2)#(b,c,num_h_patch,num_w_patch,ps[0],ps[1]) (1,2,9,2,2)
x_unfolded = rearrange(x_unfolded, 'b c n h w -> b (c h w) n')#(b, c*ps[0]*ps[1], num_patch)(1,2*4,9)
x = F.fold(x_unfolded, output_size=(new_h, new_w), kernel_size=ps, stride=ps[0])