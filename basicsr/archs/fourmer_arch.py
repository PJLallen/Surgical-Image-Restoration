"""
"""

# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from basicsr.utils.registry import ARCH_REGISTRY

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out



class InvBlock(nn.Module):
    def __init__(self, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = UNetConvBlock(self.split_len2, self.split_len1)
        self.G = UNetConvBlock(self.split_len1, self.split_len2)
        self.H = UNetConvBlock(self.split_len1, self.split_len2)

        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x):
        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out



class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = InvBlock(nc,nc//2)

    def forward(self, x):
        return x+self.block(x)



class FreBlockSpa(nn.Module):
    def __init__(self, nc):
        super(FreBlockSpa, self).__init__()
        self.processreal = nn.Sequential(
            nn.Conv2d(nc,nc,kernel_size=3,padding=1,stride=1,groups=nc),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc,nc,kernel_size=3,padding=1,stride=1,groups=nc))
        self.processimag = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1, groups=nc),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1, groups=nc))

    def forward(self,x):
        real = self.processreal(x.real)
        imag = self.processimag(x.imag)
        x_out = torch.complex(real, imag)

        return x_out


class FreBlockCha(nn.Module):
    def __init__(self, nc):
        super(FreBlockCha, self).__init__()
        self.processreal = nn.Sequential(
            nn.Conv2d(nc,nc,kernel_size=1,padding=0,stride=1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc,nc,kernel_size=1,padding=0,stride=1))
        self.processimag = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1))

    def forward(self,x):
        real = self.processreal(x.real)
        imag = self.processimag(x.imag)
        x_out = torch.complex(real, imag)

        return x_out


class SpatialFuse(nn.Module):
    def __init__(self, in_nc):
        super(SpatialFuse,self).__init__()
        # self.fpre = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlockSpa(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc,in_nc,3,1,1)
        self.cat = nn.Conv2d(2*in_nc,in_nc,3,1,1)


    def forward(self, x):
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        x_freq_spatial = self.frequency_spatial(x_freq_spatial)
        xcat = torch.cat([x,x_freq_spatial],1)
        x_out = self.cat(xcat)

        return x_out+xori


class ChannelFuse(nn.Module):
    def __init__(self, in_nc):
        super(ChannelFuse,self).__init__()
        # self.fpre = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlockCha(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc,in_nc,1,1,0)
        self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,0)


    def forward(self, x):
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        x_freq_spatial = self.frequency_spatial(x_freq_spatial)
        xcat = torch.cat([x,x_freq_spatial],1)
        x_out = self.cat(xcat)

        return x_out+xori


class ProcessBlock(nn.Module):
    def __init__(self, nc):
        super(ProcessBlock, self).__init__()
        self.spa = SpatialFuse(nc)
        self.cha = ChannelFuse(nc)

    def forward(self,x):
        x = self.spa(x)
        x = self.cha(x)

        return x


class ProcessNet(nn.Module):
    def __init__(self, nc):
        super(ProcessNet,self).__init__()
        self.conv0 = nn.Conv2d(nc, nc, 3, 1, 1)
        self.process = nn.Sequential(*[ProcessBlock(nc) for i in range(10)])
        self.convout = nn.Conv2d(nc, nc, 3, 1, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.process(x)
        xout = self.convout(x)

        return xout


@ARCH_REGISTRY.register()
class fourmer(nn.Module):
    def __init__(self, nc=32):
        super(fourmer,self).__init__()
        self.extract =  nn.Conv2d(3,nc,1,1,0)
        self.process = ProcessNet(nc)
        self.recons = nn.Conv2d(nc,3,1,1,0)

    def forward(self, x):

        x_f = self.extract(x)
        x_f = self.process(x_f)+x_f
        y = self.recons(x_f)

        return y
# net = Fourmer(32)
# # x = torch.randn((1,3,128,128))
# # print(net(x))
# print(sum(p.numel() for p in net.parameters() if p.requires_grad)/(1e6))