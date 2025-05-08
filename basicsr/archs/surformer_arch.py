## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from natten import NeighborhoodAttention2D as NeighborhoodAttention

from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


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
        return x / torch.sqrt(sigma + 1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

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


class HighOrderRegionalAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(HighOrderRegionalAttention, self).__init__()
        self.dim = dim
        self.first_region_attn = NeighborhoodAttention(dim,
                                                       kernel_size=7,
                                                       dilation=None,
                                                       num_heads=num_heads,
                                                       qkv_bias=True,
                                                       qk_scale=None,
                                                       attn_drop=0.0,
                                                       proj_drop=0.0)
        self.second_region_attn = NeighborhoodAttention(dim,
                                                        kernel_size=7,
                                                        dilation=None,
                                                        num_heads=num_heads,
                                                        qkv_bias=True,
                                                        qk_scale=None,
                                                        attn_drop=0.0,
                                                        proj_drop=0.0, )
        self.third_region_attn = NeighborhoodAttention(dim,
                                                       kernel_size=7,
                                                       dilation=None,
                                                       num_heads=num_heads,
                                                       qkv_bias=True,
                                                       qk_scale=None,
                                                       attn_drop=0.0,
                                                       proj_drop=0.0)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x, attn = self.first_region_attn(x, None)  # (1,1,h,w,kernel_size)
        x, attn = self.second_region_attn(x, attn)
        x, attn = self.third_region_attn(x, attn)
        x = x.permute(0, 3, 1, 2)
        return x


class FreBlockSpa(nn.Module):
    def __init__(self, nc):
        super(FreBlockSpa, self).__init__()
        self.processreal = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1, groups=nc),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1, groups=nc))
        self.processimag = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1, groups=nc),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1, groups=nc))

    def forward(self, x):
        real = self.processreal(x.real)
        imag = self.processimag(x.imag)
        x_out = torch.complex(real, imag)

        return x_out


class SpatialFourierModeling(nn.Module):
    def __init__(self, dim):
        super(SpatialFourierModeling, self).__init__()
        self.frequency_process = FreBlockSpa(dim)
        self.frequency_spatial = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        x_freq_spatial = self.frequency_spatial(x_freq_spatial)
        return x_freq_spatial


class ChannelWiseFourierModeling(nn.Module):
    def __init__(self, in_channels):
        super(ChannelWiseFourierModeling, self).__init__()

        # Depthwise convolution layer to process Fourier components
        self.dw1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, groups=in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: Input feature map of shape (B, C, H, W)
        B, C, H, W = x.shape

        # Global Average Pooling (GAP)
        gap = x.mean(dim=(2, 3))  # Shape: (B, C)

        # Fourier Transform along the channel dimension
        fourier_transformed = torch.fft.fft(gap, dim=1)  # Shape: (B, C)
        FR = fourier_transformed.real
        FI = fourier_transformed.imag

        # Apply depthwise convolution and ReLU to real and imaginary parts
        CR = self.relu(self.dw1(FR.unsqueeze(-1)).squeeze(-1))  # Shape: (B, C)
        CI = self.relu(self.dw1(FI.unsqueeze(-1)).squeeze(-1))  # Shape: (B, C)

        # Reconstruct using Inverse Fourier Transform (IFT)
        reconstructed = torch.fft.ifft(CR + 1j * CI, dim=1).real  # Shape: (B, C)

        # Expand back to spatial dimensions
        reconstructed = reconstructed.view(B, C, 1, 1).expand(B, C, H, W)  # Shape: (B, C, H, W)

        return reconstructed


class FourierRegionalTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(FourierRegionalTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.high_order_attn = HighOrderRegionalAttention(dim,
                                                          num_heads=num_heads)
        self.spatial_fourier_modeling = SpatialFourierModeling(dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.norm4 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.channel_fourier_modeling = ChannelWiseFourierModeling(dim)

    def forward(self, x):
        x = x + self.high_order_attn(self.norm1(x))
        x = x + self.spatial_fourier_modeling(self.norm2(x))
        x = x + self.ffn(self.norm3(x))
        x = x + self.channel_fourier_modeling(self.norm4(x))
        return x


##########################################################################

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


import random


##########################################################################
##---------- Restormer -----------------------
@ARCH_REGISTRY.register()
class surformer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[2, 3, 3, 4],  # 2334#
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(surformer, self).__init__()
        self.dim = dim
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            FourierRegionalTransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias,
                                            LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            FourierRegionalTransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1],
                                            ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            FourierRegionalTransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                                            ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            FourierRegionalTransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3],
                                            ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            FourierRegionalTransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                                            ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            FourierRegionalTransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1],
                                            ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            FourierRegionalTransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0],
                                            ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            FourierRegionalTransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0],
                                            ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in
            range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.aux_branch = nn.Sequential(*[
            FourierRegionalTransformerBlock(dim=int(dim * 2), num_heads=heads[0],
                                            ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(3)],
                                        nn.Conv2d(dim * 2, dim, 1, 1, 0))
        self.feature_align_latent = nn.Conv2d(dim * 2 ** 3, dim, 1, 1, 0)
        self.feature_align_last = nn.Conv2d(dim * 2, dim, 1, 1, 0)
        self.sam_img_conv = nn.Conv2d(3, dim, 3, 1, 1)

    def forward(self, inp_img, sam_img=None):

        inp_enc_level1 = self.patch_embed(inp_img)  # 卷积提取特征
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        sample_first = out_enc_level1
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        sample_mid = latent
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        sample_last = out_dec_level1
        out_dec_level1 = self.refinement(out_dec_level1)
        random_sample_list = [sample_first, sample_mid, sample_last]
        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img
        if self.training:
            # Set the masking probability
            p = 0.5  # Probability of keeping a position
            sampled_feature = random.choice(random_sample_list)
            b, c, h, w = sampled_feature.shape
            if c == self.dim:
                pass
            elif c == self.dim * 2:
                sampled_feature = self.feature_align_last(sampled_feature)
            else:
                sampled_feature = self.feature_align_latent(sampled_feature)
            # Generate Bernoulli mask with probability p
            mask = torch.bernoulli(torch.full_like(sampled_feature, p))  # Binary mask ∼ Bernoulli(p)
            # Apply the mask to the features
            masked_feature = sampled_feature * mask
            sam_fea = self.sam_img_conv(sam_img)
            sam_fea = F.interpolate(sam_fea, size=(h, w))
            # Concatenate masked features with semantic features
            aux_input = torch.cat([masked_feature, sam_fea], dim=1)
            # Pass through aux_branch
            aux_output = self.aux_branch(aux_input)
            # print(aux_output.shape)
            # Calculate reconstruction loss
            aux_loss = torch.nn.functional.mse_loss(aux_output, sampled_feature)
            return out_dec_level1, aux_loss
        else:
            # print(out_dec_level1.shape)
            return out_dec_level1
        # return out_dec_level1


if __name__ == '__main__':
    net = surformer().to('cuda')
    inp = torch.randn((1, 3, 64, 64)).to('cuda')
    print(net(inp, inp))
    print(sum(p.numel() for p in net.parameters()))