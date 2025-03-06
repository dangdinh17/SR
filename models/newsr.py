import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out
import math
from einops import rearrange
import warnings
from torchsummary import summary
#building CBAM, SpatialAttention, ChannelAttention, SelfAttention, 
#Window multihead self attention

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor # type: ignore
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_rate=16):
        super(ChannelAttention, self).__init__()
        self.pool = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1)
        ])
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_rate, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction_rate, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # perform pool with independent Pooling
        b, c, _, _ = x.size()
        avg_feat = self.pool[0](x).view(b, c)
        max_feat = self.pool[1](x).view(b, c)
        # perform mlp with the same mlp sub-net
        avg_out = self.mlp(avg_feat)
        max_out = self.mlp(max_feat)
        # attention
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return attention * x #B , C, H, W
    
    
class SpatialAttention(nn.Module):
    def __init__(self, channels, reduction_rate=16):
        super(SpatialAttention, self).__init__()
        self.pool = nn.ModuleList([
            nn.AdaptiveMaxPool2d(1),
            nn.AdaptiveAvgPool2d(1)
        ])
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_rate, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction_rate, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        out = self.pool[0](x)
        out = self.pool[1](out)
        attention = self.mlp(out.view(b, c)).view(b, c, 1, 1)
        return x * attention #B , C, H, W
    
class ATTB(nn.Module):
    def __init__(self, channels, groups = 'depthwise', bias=False):
        super().__init__()
        self.channel_att = ChannelAttention(channels=channels)
        self.spatial_att = SpatialAttention(channels=channels)
        
        self.norm = nn.LayerNorm(channels)
        if groups == 'depthwise':
            groups = channels
        elif groups == 'standard':
            groups = 1
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=bias),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=bias)
        )
        
    def forward(self, x):
        b, c, h, w = x.size()
        input = self.norm(x.reshape(b, -1, c)).reshape(b, c, h, w)
        chan_out = self.channel_att(input)
        spar_out = self.spatial_att(input)
        output = self.mlp(chan_out + spar_out)
        return x + output
  
class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels//chan_factor), 1, stride=1, padding=0, groups=in_channels//chan_factor, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
            )

    def forward(self, x):
        return self.bot(x)
    
class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels*chan_factor), kernel_size=1, stride=1, padding=0, groups=in_channels, bias=bias)
            )

    def forward(self, x):
        return self.bot(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class PA(nn.Module):
    def __init__(self, channels=64, chanfactor=4, groups='depthwise', bias=False):
        super().__init__()
        if groups == 'depthwise':
            groups = channels
        elif groups == 'standard':
            groups = 1
        self.channels = channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Sigmoid()
        )
       
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=bias)
        
    def forward(self,x):
        b, c, h, w = x.shape
        
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        
        out = out1 * out2
        return out

class SCPA(nn.Module):
    def __init__(self, channels=64, chanfactor=4, groups='depthwise', bias=False):
        super().__init__()
        if groups == 'depthwise':
            groups = channels
        elif groups == 'standard':
            groups = 1
        self.channels = channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels//chanfactor, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.GELU(),
            PA(channels=channels//chanfactor),
            nn.Conv2d(channels//chanfactor, channels//chanfactor, kernel_size=3, stride=1, padding=1, groups=groups//chanfactor, bias=bias)
        )
       
        self.conv2 = nn.Sequential(
                    nn.Conv2d(channels, channels//chanfactor, kernel_size=1, stride=1, padding=0, bias=bias),
                    nn.GELU(),
                    nn.Conv2d(channels//chanfactor, channels//chanfactor, kernel_size=3, stride=1, padding=1, groups=groups//chanfactor, bias=bias)
        )
        self.fuse = nn.Conv2d(2*channels//chanfactor, channels, kernel_size=1, stride=1, padding=0, groups=groups//chanfactor, bias=bias)
        
        
    def forward(self,x):
        b, c, h, w = x.shape
        
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        
        out = torch.cat([out1, out2], dim=1)
        # print(f"Shape after conv1: {out1.shape}, Shape after conv2: {out2.shape}, "
        #         f"Shape after concatenation: {out.shape}, Shape after fusion: {out.shape}")        
        out = self.fuse(out)
        return out + x
    
class SKM(nn.Module):
    def __init__(self, channels=64, chanfactor=4, groups='depthwise', bias=False):
        super().__init__()
        if groups == 'depthwise':
            groups = channels
        elif groups == 'standard':
            groups = 1
        self.channels = channels
        self.path1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=bias),
            ChannelAttention(channels),
            SpatialAttention(channels)
        )
        self.path3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=bias),
            ChannelAttention(channels),
            SpatialAttention(channels)
        )
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=bias)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels//chanfactor),
            nn.ReLU(),
            nn.Linear(channels//chanfactor, channels*2),
            nn.Sigmoid()
            )
        
    def forward(self,x):
        b, c, h, w = x.shape
        path1 = self.path1(x)
        path3 = self.path3(x)
        
        path2 = self.conv(x)
        path2 = self.pool(path2).view(b, c)
        path2 = self.mlp(path2)
        path21, path23 = path2[:, :self.channels], path2[:, self.channels:]

        path21 = path21.view(b, c, 1, 1)
        path23 = path23.view(b, c, 1, 1)
        out = path1 * path21 + path3 * path23
        return out


    
class NormMLP(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        self.fn =  nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: B C H W
        out: B C H W
        """
        B, C, H, W = x.shape
        x = self.norm(x.view(B, C, -1).permute(0, 2, 1))
        x = x.reshape(B, C, H, W)
        out = self.fn(x)
        return out
    
class MHSA(nn.Module):
    def __init__(self, embed_dim,dim=64, num_heads=4):
        """
        Multi-Head Self-Attention (MHSA).
        Args:
            embed_dim (int): Số chiều embedding (bằng số channels).
            num_heads (int): Số lượng heads trong Multi-head Attention.
            dropout (float): Xác suất dropout.
        """
        super(MHSA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Layers
        self.norm = nn.LayerNorm(embed_dim)
        self.to_q = nn.Linear(embed_dim, dim * num_heads, bias=False)
        self.to_k = nn.Linear(embed_dim, dim * num_heads, bias=False)
        self.to_v = nn.Linear(embed_dim, dim * num_heads, bias=False)
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim*num_heads, embed_dim)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False, groups=embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False, groups=embed_dim),
        )
        self.dim = dim
    def forward(self, x):
        """
        x: B C H W
        out: B C H W
        """
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C), N = H * W
        x = self.norm(x)  # Áp dụng LayerNorm

        # Tính toán Q, K, V
        # qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads)  # (B, N, 3, num_heads, head_dim)
        # qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # q, k, v: (B, num_heads, N, head_dim)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                         (q_inp, k_inp, v_inp))
        
        # print(q.shape)
        q = q.transpose(-2, -1)  # (b, heads, dim_head, hw)
        k = k.transpose(-2, -1)  # (b, heads, dim_head, hw)
        v = v.transpose(-2, -1)  # (b, heads, dim_head, hw)
        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = self.softmax(attn)  # (B, num_heads, N, N)

        # Tính giá trị (Value)
        out = (attn @ v)
        
        out = out.permute(0, 3, 1, 2).reshape(B, H*W, self.dim*self.num_heads)  # (B, N, C)
        # print(out.shape)
        out = self.proj(out).view(B, C, H, W) # (B, N, C)
        out = self.pos_emb(out)

        return out
    
class TAB(nn.Module):
    def __init__(self, embed_dim, num_blocks=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MHSA(embed_dim=embed_dim),
                NormMLP(dim=embed_dim)
            ]))
    def forward(self, x):
        for (mhsa, ff) in self.blocks:
            x = mhsa(x) + x
            x = ff(x) + x
        out = x
        return out
        
#IluminationExtractFeature
class EFM(nn.Module):
    def __init__(self, midd_chan=64, num_blocks=[4, 2, 1]):
        super().__init__()
        
        self.att = ATTB(midd_chan)
        self.down = nn.ModuleList([
            DownSample(midd_chan, 2, 2),
            DownSample(midd_chan*2, 2, 2)
        ])
        self.up = nn.ModuleList([
            UpSample(midd_chan*2, 2, 2),
            UpSample(midd_chan*4, 2, 2)
        ])
        self.path = nn.ModuleList([])
        for i, num_block in enumerate(num_blocks):
            self.path.append(nn.ModuleList([
                nn.Sequential(*[SCPA(midd_chan*(2**i)) for _ in range(num_block)]),
                TAB(midd_chan*(2**i), num_block)
            ]))
        self.kf1 = SKM(midd_chan*2)
        self.kf0 = SKM(midd_chan)
        
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        #3 branch downsample
        
        inp = self.att(x)
        br0 = inp
        br1 = self.down[0](inp)
        br2 = self.down[1](br1)
        
        #SCPA process
        scpa0 = self.path[0][0](br0)
        scpa1 = self.path[1][0](br1)
        scpa2 = self.path[2][0](br2)
        # print(f"scpa0 shape: {scpa0.shape}, scpa1 shape: {scpa1.shape}, scpa2 shape: {scpa2.shape}")
        #SKM process
        scpa2 = self.path[2][1](scpa2)
        scpa2 = self.up[1](scpa2)
        # print(scpa1.shape, scpa2.shape)
        sk1 = scpa1 + scpa2
        
        sk1 = self.kf1(sk1)
        sk1 = self.path[1][1](sk1) 
        sk1 = self.up[0](sk1)
        
        sk0 = scpa0 + sk1
        sk0 = self.path[0][1](sk0)
        #fuse WMSA
        output = x + sk0
        return output

class CISR(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, scale=4, channels=64, num_ief=3, num_blocks=[1, 1, 1]):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        # Sử dụng nn.ModuleList thay vì nn.Sequential
        self.ief = nn.ModuleList([EFM(midd_chan=channels, num_blocks=num_blocks) for _ in range(num_ief)])
        self.out = nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if scale == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=channels*4, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.Conv2d(in_channels=channels, out_channels=channels*4, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
            )
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=channels*scale**2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(scale),
            )
    def forward(self, x):
        x = self.conv(x)
        out=x
        for module in self.ief:  # Duyệt qua các khối IluminationExtractFeature
            out = module(out)
        out = out + x
        
        out = self.upscale(out)
        out = self.out(out)
        # print(out.shape)
        return out

# if __name__ == '__main__':
#     from fvcore.nn import FlopCountAnalysis
#     model = Model(num_ief=1, channels=64, num_blocks=[4, 2, 1])
#     # print(model)
#     inputs = torch.randn((1, 3, 510, 339))
#     # summary(model, (3, 64, 64))
#     flops = FlopCountAnalysis(model, inputs)
#     # model = TAB(embed_dim=64, num_blocks=4)  # Thay thế bằng khối bạn muốn tính toán
#     # num_params = sum(p.numel() for p in model.parameters() )
#     # print(f"Number of parameters: {num_params}")
#     n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量
#     print(f'GMac:{flops.total()/(1024*1024*1024)}')
#     print(f'Params:{n_param}')
    