import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2  # 仅用于图像加载和显示，核心计算在 PyTorch 中
import pywt  # 小波变换库
from functools import partial
import math

def gaussian_blur_torch(image, kernel_size):
    """使用 PyTorch 实现高斯模糊。"""
    pad = (kernel_size - 1) // 2
    kernel = _gaussian_kernel2d(kernel_size, sigma=kernel_size / 6).float().to(image.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(image.size(1), 1, 1, 1)
    return F.conv2d(image, kernel, padding=pad, groups=image.size(1))

def _gaussian_kernel2d(kernel_size, sigma):
    x_coords = torch.arange(kernel_size)
    x_coords -= (kernel_size - 1) // 2
    y_coords = x_coords.unsqueeze(0).t()
    kernel = torch.exp(-(x_coords**2 + y_coords**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

class DownsampleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2, padding=1):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.upsample(x))

class FourierCombineNet(nn.Module):
    def __init__(self, dim, gaussian_kernel_size=5):
        super().__init__()
        self.gaussian_kernel_size = gaussian_kernel_size

        # 编码器 (3层下采样卷积)
        self.low_freq_encoder = nn.Sequential(
            DownsampleConvBlock(dim, dim),
            DownsampleConvBlock(dim, dim),
            DownsampleConvBlock(dim, dim)
        )
        self.high_freq_encoder = nn.Sequential(
            DownsampleConvBlock(dim, dim),
            DownsampleConvBlock(dim, dim),
            DownsampleConvBlock(dim, dim)
        )

        # 解码器 (3层上采样卷积)
        self.low_freq_decoder = nn.Sequential(
            UpsampleConvBlock(dim, dim),
            UpsampleConvBlock(dim, dim),
            UpsampleConvBlock(dim, dim)
        )
        self.high_freq_decoder = nn.Sequential(
            UpsampleConvBlock(dim, dim),
            UpsampleConvBlock(dim, dim),
            UpsampleConvBlock(dim, dim)
        )

    def forward(self, x):
        # 1. 高斯低通滤波获取低频信号 (空间域)
        low_freq_spatial = gaussian_blur_torch(x, self.gaussian_kernel_size)

        # 2. 高通滤波获取高频信号 (空间域)
        high_freq_spatial = x - low_freq_spatial

        # 3. 对低频信号进行编码和解码 (空间域)
        low_freq_encoded = self.low_freq_encoder(low_freq_spatial)
        low_freq_decoded_spatial = self.low_freq_decoder(low_freq_encoded)

        # 4. 对高频信号进行编码和解码 (空间域)
        high_freq_encoded = self.high_freq_encoder(high_freq_spatial)
        high_freq_decoded_spatial = self.high_freq_decoder(high_freq_encoded)

        # 5. 对解码后的高低频信号分别做傅里叶逆变换
        # PyTorch 的 FFT 操作需要处理复数，并且通常在最后一个维度进行
        low_freq_fft = torch.fft.fft2(torch.complex(low_freq_decoded_spatial, torch.zeros_like(low_freq_decoded_spatial)))
        low_freq_reconstructed_spatial = torch.fft.ifft2(low_freq_fft).real

        high_freq_fft = torch.fft.fft2(torch.complex(high_freq_decoded_spatial, torch.zeros_like(high_freq_decoded_spatial)))
        high_freq_reconstructed_spatial = torch.fft.ifft2(high_freq_fft).real

        # 6. 将逆变换的结果相加
        reconstructed_image = low_freq_reconstructed_spatial + high_freq_reconstructed_spatial

        return reconstructed_image

    
def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size//4, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    # x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    # x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class WaveletAttention(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, patch_size=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.patch_size = patch_size

        # 前处理：降低到1/4通道
        self.pre_conv = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        reduced_c = in_channels // 4
        embed_dim = reduced_c*patch_size*patch_size
        self.embed_dim = embed_dim

        # patch embedding
        # self.patch_embed = nn.Conv2d(reduced_c, reduced_c, kernel_size=patch_size, stride=patch_size)
        self.patch_embed_x = nn.Sequential(
            nn.Conv2d(reduced_c, embed_dim//4, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(embed_dim//4),
            nn.GELU(),
            nn.Conv2d(embed_dim//4, embed_dim, kernel_size=1)  # 升维
        )
        self.patch_embed_h = nn.Sequential(
            nn.Conv2d(reduced_c, embed_dim//4, kernel_size=patch_size//2, stride=patch_size//2),
            nn.BatchNorm2d(embed_dim//4),
            nn.GELU(),
            nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=1)  # 升维
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim//4)

        # q, k, v 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim//4, embed_dim)
        self.v_proj = nn.Linear(embed_dim//4, embed_dim)

        # 小波分支
        self.h_conv = nn.Sequential(
            nn.Conv2d(reduced_c * 4, reduced_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(reduced_c),
            nn.GELU(),
        )

        # 多头注意力
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

        self.wt_filter, self.iwt_filter = create_wavelet_filter('db1', reduced_c, reduced_c, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        # 上采样层（恢复回原尺寸）
        self.upsample = nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False)

        # 输出映射
        self.out_conv = nn.Conv2d(reduced_c*2, in_channels, kernel_size=1)

    def forward(self, x):

        # --- 前处理 ---
        x = self.pre_conv(x)  # [B, C//4, H, W]
        B, C, H, W = x.shape
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size

        # --- patch embedding下采样 ---
        x_patch = self.patch_embed_x(x)  # [B, emb, H/4, W/4]

        # --- q映射 ---
        x_flat = x_patch.flatten(2).permute(0,2,1)  # [HW', B, emb]
        x_flat = self.layernorm1(x_flat)
        q = self.q_proj(x_flat)                       # [HW', B, embed_dim]

        # x_wave = torch.cat([LL, LH, HL, HH], dim=1)    # [B, 4*C_reduced, H/2, W/2]
        x_wave = self.wt_function(x)
        h = self.h_conv(x_wave)                        # [B, C_reduced, H/2, W/2]

        h_patch = self.patch_embed_h(h)
        # --- k,v映射 ---
        h_flat = h_patch.flatten(2).permute(0,2,1)          # [HW', B, C_reduced]
        h_flat = self.layernorm2(h_flat)
        k = self.k_proj(h_flat)                         # [HW', B, embed_dim]
        v = self.v_proj(h_flat)                         # [HW', B, embed_dim]

        # --- 多头注意力 ---
        attn_out, _ = self.attention(q, k, v)            # [HW', B, embed_dim]

        # --- 恢复成图像 ---
        attn_out = attn_out.reshape(B, num_patches_h, num_patches_w, C, self.patch_size, self.patch_size)
        attn_out = attn_out.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        # attn_out = attn_out.permute(1, 2, 0)             # [B, embed_dim, HW']
        # H_patch, W_patch = x_patch.shape[2], x_patch.shape[3]
        # attn_out = attn_out.view(B, self.embed_dim, H_patch, W_patch)

        # --- 上采样 ---
        # attn_out = self.upsample(attn_out)               # [B, embed_dim, H, W]

        # --- 小波逆变换恢复 h ---
        h_recon = self.iwt_function(h)         # [B*C_reduced, H, W]
        # h_recon = h_recon.view(B, C_reduced, H, W)

        # --- 拼接 ---
        out = torch.cat([attn_out, h_recon], dim=1)      # [B, embed_dim+C_reduced, H, W]

        # --- 输出映射 ---
        out = self.out_conv(out)                         # [B, in_channels, H, W]

        return out


# CBAM模块的实现
class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 先应用通道注意力
        channel_att = self.channel_attention(x)
        x = x * channel_att
        # 然后应用空间注意力
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att
        return x

# 图像块分割
def extract_patches(x, patch_size):
    b, c, h, w = x.size()
    patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(b, c, -1, patch_size, patch_size)
    return patches

# 主网络
class PatchFFT(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, layer=8):
        super(PatchFFT, self).__init__()
        pooled_chn = in_channels // 2**(layer+2)
        self.pooled_chn = pooled_chn
        patch_size = 2**(4-layer)
        embed_dim = pooled_chn*patch_size*patch_size
        self.embed_dim = pooled_chn*patch_size*patch_size
        self.num_heads = num_heads
        self.patch_size = patch_size 
        self.pre_process = nn.Sequential(
            nn.Conv2d(in_channels, pooled_chn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(pooled_chn),
            nn.ReLU(inplace=True)
        )
        # Patch Embedding使用1x1卷积
        # self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(pooled_chn, embed_dim//4, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(embed_dim//4),
            nn.GELU(),
            nn.Conv2d(embed_dim//4, embed_dim, kernel_size=1)  # 升维
        )
        self.layernorm = nn.LayerNorm(embed_dim)
        # self.cbam = CBAM(embed_dim)

        # 1x1卷积层
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, padding=0)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        # 多头自注意力
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)

        # 输出卷积，恢复特征图的尺寸
        self.output_conv = nn.Sequential(
            nn.Conv2d(pooled_chn, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # res = x.clone()
        # 获取输入图像的尺寸
        x = self.pre_process(x)
        b, c, h, w = x.size()
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size

        # 1. 图像块分割
        patches = extract_patches(x, self.patch_size)  # (b, c, n_patches, patch_size, patch_size)
        patches = patches.reshape(b, c, -1, self.patch_size * self.patch_size).transpose(1, 2)  # (b, n_patches, c * patch_size * patch_size)

        # 2. 对每个patch应用傅里叶变换
        freq_patches = torch.fft.fft2(patches.reshape(-1, c, self.patch_size, self.patch_size))
        freq_patches = torch.view_as_real(freq_patches)[:,:,:,:,0]  # 16,3,8,8
        
        # 3. Patch embedding (卷积层)
        embedded_patches = self.patch_embed(freq_patches)
        
        # 4. 1x1卷积 + RELU + 1x1卷积
        x = self.conv1(embedded_patches)
        x = F.relu(x)
        x = self.conv2(x)

        # 5. CBAM模块处理
        # x = self.cbam(x)

        # 6. 多头自注意力
        x = x.reshape(b, -1, self.embed_dim)  # 转换为(batch_size, seq_len=, embed_dim)的形状
        x = self.layernorm(x)
        x, _ = self.mha(self.query(x), self.key(x), self.value(x))

        # 7. 将图像块重新组合成图像特征
        # x = x.reshape(b, c, h, w)  # 恢复为(b, embed_dim, h, w)
        x = x.reshape(b, num_patches_h, num_patches_w, c, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)

        # 8. 输出卷积层恢复通道
        output = self.output_conv(x)

        return output


class FrequencySeparation(nn.Module):
    def __init__(self, in_channels, height, width):
        super(FrequencySeparation, self).__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width

        # 定义两个可学习的频域掩码（用于高频和低频）
        # 掩码大小为 [1, C, H, W, 2]，最后一维是实部和虚部（作为复数乘子）
        self.high_mask = nn.Parameter(torch.randn(1, in_channels, height, width, 2) * 0.01)
        self.low_mask = nn.Parameter(torch.randn(1, in_channels, height, width, 2) * 0.01)
        # self.high_mask_real = nn.Parameter(torch.zeros(1, in_channels, height, width))
        # self.high_mask_imag = nn.Parameter(torch.zeros(1, in_channels, height, width))
        # self.low_mask_real = nn.Parameter(torch.ones(1, in_channels, height, width))
        # self.low_mask_imag = nn.Parameter(torch.zeros(1, in_channels, height, width))
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        assert h == self.height and w == self.width, "Input size must match initialized size"

        # 1. 傅里叶变换 -> [B, C, H, W] -> [B, C, H, W], complex
        freq = torch.fft.fft2(x, norm='ortho')  # complex tensor, shape: [B, C, H, W]

        # 2. 生成复数掩码
        high_mask_complex = torch.view_as_complex(self.high_mask)  # [1, C, H, W]
        low_mask_complex = torch.view_as_complex(self.low_mask)    # [1, C, H, W]

        # 3. 应用高低频掩码（频域滤波）
        freq_high = freq * high_mask_complex
        freq_low = freq * low_mask_complex

        # 4. 逆傅里叶变换 -> [B, C, H, W], complex -> real
        img_high = torch.fft.ifft2(freq_high, norm='ortho').real  # 取实部
        img_low = torch.fft.ifft2(freq_low, norm='ortho').real
        new_freq = torch.concat([img_high, img_low], dim=1)
        out = self.output_conv(new_freq)
        

        return out
    


class Freq_block(nn.Module):
    def __init__(self, dim,dfilter_freedom=[3, 2],
                 dfilter_type='piecewise_linear'):
        super().__init__()
        self.dim = dim
        self.dw_amp_conv = nn.Sequential(
            nn.Conv2d(dim, dim, groups=dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.df1 = nn.Sequential(
            nn.Conv2d(2, 2, groups=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.df2 = nn.Sequential(
            nn.Conv2d(2, 2, groups=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.dw_pha_conv = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, groups=dim*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
            )

    def forward(self, x):
        b,c,h,w = x.shape
        msF = torch.fft.rfft2(x+1e-8, dim=(-2, -1))
        msF = torch.cat([
            msF[:, :, msF.size(2) // 2 + 1:, :],
            msF[:, :, :msF.size(2) // 2 + 1, :]], dim=2)
        # msF = torch.fft.fftshift(msF, dim=(-2, -1))
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)

        amp_fuse = self.dw_amp_conv(msF_amp)
        avg_attn = torch.mean(amp_fuse, dim=1, keepdim=True)
        max_attn, _ = torch.max(amp_fuse, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        agg=self.df1(agg)
        amp_fuse=amp_fuse*agg
        amp_res = amp_fuse - msF_amp
        pha_guide=torch.cat((msF_pha,amp_res),dim=1)
        pha_fuse = self.dw_pha_conv(pha_guide)
        avg_attn = torch.mean(pha_fuse, dim=1, keepdim=True)
        max_attn, _ = torch.max(pha_fuse, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        agg = self.df2(agg)
        pha_fuse = pha_fuse * agg
        pha_fuse=pha_fuse*(2.*math.pi)-math.pi
        # pha_fuse = torch.clamp(pha_fuse, -math.pi, math.pi)
        ## amp_fuse = amp_fuse + msF_amp
        # pha_fuse = pha_fuse + msF_pha

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)
        # out=torch.fft.ifftshift(out, dim=(-2, -1))
        out = torch.cat([
            out[:, :, out.size(2) // 2 - 1:, :],
            out[:, :, :out.size(2) // 2 - 1, :]], dim=2)
        out = torch.abs(torch.fft.irfft2(out+1e-8, s=(h, w)))
        if torch.isnan(out).sum()>0:
            print('freq feature include NAN!!!!')
            assert torch.isnan(out).sum() == 0
            out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
        out = out + x
        return F.relu(out)


# ... 文件前面的代码保持不变 ...

if __name__ == "__main__":
    # 测试参数
    batch_size = 2
    channels = 128
    height = 64
    width = 64
    
    # 创建测试输入和模型
    test_input = torch.randn(batch_size, channels, height, width)
    # model = FrequencySeparation(in_channels=channels, height=height, width=width)
    # model = WaveletAttention(
    #     in_channels=channels,
    #     embed_dim=64,
    #     num_heads=4,
    #     patch_size=4
    # )
    model = Freq_block(dim=channels)
    
    # 前向传播测试
    output = model(test_input)
    
    # 打印结果
    print(f"输入尺寸: {test_input.shape}")
    print(f"输出尺寸: {output.shape}")
    print("测试通过，输入输出尺寸匹配")