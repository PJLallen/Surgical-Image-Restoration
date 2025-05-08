
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from thop import profile, clever_format
def pad_input(input_tensor):
    b,c,h,w = input_tensor.shape
    pad_h = (64-h%64)%64
    pad_w =  (64-w%64)%64
    padding = (0,pad_w,0,pad_h)
    padded_tensor = F.pad(input_tensor,padding,mode='constant',value=0)
    return padded_tensor,(h,w)
def unpad(padded_tensor,original_size):
    h,w = original_size
    return padded_tensor[:,:,:h,:w]
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            DeepPoolLayer(in_channel, out_channel) if filter else nn.Identity(),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class DeepPoolLayer(nn.Module):
    def __init__(self, k, k_out):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [8, 4, 2]
        dilation = [3, 7, 9]
        pools, convs, dynas = [], [], []
        for j, i in enumerate(self.pools_sizes):
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
            dynas.append(MultiShapeKernel(dim=k, kernel_size=3, dilation=dilation[j]))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.dynas = nn.ModuleList(dynas)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            if i == 0:
                y = self.dynas[i](self.convs[i](self.pools[i](x)))
            else:
                y = self.dynas[i](self.convs[i](self.pools[i](x) + y_up))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_sizes) - 1:
                y_up = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        resl = self.relu(resl)
        resl = self.conv_sum(resl)

        return resl


class dynamic_filter(nn.Module):
    def __init__(self, inchannels, kernel_size=3, dilation=1, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        self.dilation = dilation

        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)
        self.act = nn.Tanh()

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.pad = nn.ReflectionPad2d(self.dilation * (kernel_size - 1) // 2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.inside_all = nn.Parameter(torch.zeros(inchannels, 1, 1), requires_grad=True)

    def forward(self, x):
        identity_input = x
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)

        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size, dilation=self.dilation).reshape(n, self.group,
                                                                                                c // self.group,
                                                                                                self.kernel_size ** 2,
                                                                                                h * w)

        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)

        low_filter = self.act(low_filter)

        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_low = low_part * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)

        out_low = out_low * self.lamb_l[None, :, None, None]

        out_high = (identity_input) * (self.lamb_h[None, :, None, None] + 1.)

        return out_low + out_high


class cubic_attention(nn.Module):
    def __init__(self, dim, group, dilation, kernel) -> None:
        super().__init__()

        self.H_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta


class spatial_strip_att(nn.Module):
    def __init__(self, dim, kernel=3, dilation=1, group=2, H=True) -> None:
        super().__init__()

        self.k = kernel
        pad = dilation * (kernel - 1) // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel // 2, 1) if H else (1, kernel // 2)
        self.dilation = dilation
        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group * kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Tanh()
        self.inside_all = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)
        self.lamb_l = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(dim), requires_grad=True)
        gap_kernel = (None, 1) if H else (1, None)
        self.gap = nn.AdaptiveAvgPool2d(gap_kernel)

    def forward(self, x):
        identity_input = x.clone()
        filter = self.ap(x)
        filter = self.conv(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel, dilation=self.dilation).reshape(n, self.group,
                                                                                           c // self.group, self.k,
                                                                                           h * w)
        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1 // self.k, self.k, p * q).unsqueeze(2)
        filter = self.filter_act(filter)
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)

        out_low = out * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)
        out_low = out_low * self.lamb_l[None, :, None, None]
        out_high = identity_input * (self.lamb_h[None, :, None, None] + 1.)

        return out_low + out_high


class MultiShapeKernel(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1, group=8):
        super().__init__()

        self.square_att = dynamic_filter(inchannels=dim, dilation=dilation, group=group, kernel_size=kernel_size)
        self.strip_att = cubic_attention(dim, group=group, dilation=dilation, kernel=kernel_size)

    def forward(self, x):
        x1 = self.strip_att(x)
        x2 = self.square_att(x)

        return x1 + x2

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res-1)]
        layers.append(ResBlock(out_channel, out_channel, filter=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res-1)]
        layers.append(ResBlock(channel, channel, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))
from basicsr.utils.registry import ARCH_REGISTRY
@ARCH_REGISTRY.register()
class ConvIR(nn.Module):
    def __init__(self, num_res=16):
        super(ConvIR, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):

        if not self.training:
            _,_,h,w = x.shape
            x,_ = pad_input(x)

        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        # 256
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        # 128
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        # 64
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        # 128
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        # 256
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)
        if not self.training:
            return unpad(outputs[-1],(h,w))
        return outputs


def build_net():
    return ConvIR()

# def test_model():
#     # 设置设备为 GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 初始化网络和输入
#     net = ConvIR().to(device)
#     x = torch.randn(1, 3, 720, 1080).to(device)
#
#     # 计算参数量
#     total_params = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
#     print(f"Total Parameters (in million): {total_params:.2f}M")
#
#     # 计算 FLOPs
#     flops, params = profile(net, inputs=(x,))
#     flops = clever_format([flops], "%.3f")
#     print(f"FLOPs: {flops}")
#
#     # 跳过第一次运行时间
#     with torch.no_grad():
#         for _ in range(10):
#             _ = net(x)
#
#     # 测量 FPS
#     with torch.no_grad():
#         start_time = time.time()
#         for _ in range(100):  # 运行100次前向传播
#             _ = net(x)
#         end_time = time.time()
#
#     # 计算 FPS
#     elapsed_time = end_time - start_time
#     fps = 100 / elapsed_time
#     print(f"FPS: {fps:.2f} frames per second")
#
#
# if __name__ == "__main__":
#     test_model()