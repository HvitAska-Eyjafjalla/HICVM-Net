# 引入 VSS Block
import numpy as np

from .vmamba import *

import torch
from thop import profile

from torch.utils.checkpoint import checkpoint


# 缩写规范：
# sff：Spectral Feature Fusion Block
# cvm：Cascaded VMamba Block
# fe: Feature Enhance Block

def calculating_params_flops(model, size):
    input = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    # 打印计算量
    print("FLOPs: %.4fG" % (flops / 1e9))
    # 打印参数量
    print("Params: %.4fM" % (params / 1e6))

    total = sum(p.numel() for p in model.parameters())
    # 打印总参数量
    print("Total params: %.4fM" % (total / 1e6))


class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class CascadedVMambaBlock(nn.Module):
    def __init__(self,
                 input_channel,
                 head,
                 hidden_state,
                 ss2d_expand_ratio,
                 drop_path_rate,
                 channel_shuffle):
        super().__init__()
        # 第一部分：参数定义
        # ——————多头参数——————
        # 定义 SS2D多头数
        self.head = head
        # 定义 输入CascadedVMambaBlock通道数
        self.input_channel = input_channel
        # 若 输入CascadedVMambaBlock通道数 不能被 SS2D多头数 整除，抛出assert异常
        assert input_channel % head == 0, 'input_channel cannot be divided by head'
        # 定义 是否启用通道乱序
        self.channel_shuffle_bool = channel_shuffle

        # ——————VSSBlock & SS2D参数——————
        # 定义 隐状态维度
        self.hidden_state = hidden_state
        # 定义 VSSBlock随机失活率
        self.VSSBlock_drop_rate = drop_path_rate
        # 定义 SS2D内部数据扩张比率
        self.SS2D_expand_ratio = ss2d_expand_ratio
        # 定义 SS2D随机失活率
        self.SS2D_drop_rate = drop_path_rate

        # 第二部分：模块初始化
        self.layerNorm = nn.LayerNorm(input_channel)
        self.channel_shuffle_module = ShuffleBlock(self.head)
        self.VSSBlock = VSSBlock(hidden_dim=self.input_channel // self.head,
                                 drop_path=self.VSSBlock_drop_rate,
                                 attn_drop_rate=self.SS2D_drop_rate,
                                 d_state=self.hidden_state,
                                 ss2d_expand_ratio=self.SS2D_expand_ratio)
        self.VSSBlock_skip_scale = nn.Parameter(torch.ones(1))
        self.CVM_skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # 指示输入x的形状格式，并获取x的形状大小
        B, H, W, C = x.shape
        # 记录原始x
        # x_original = x
        # 若启用通道乱序
        if self.channel_shuffle_bool:
            # 对张量进行变换，B H W C → B C H W
            x_transposed_BCHW = x.permute(0, 3, 1, 2).contiguous()
            # 进行目标为 SS2D多头数 群组的通道乱序重排
            x_shuffled = self.channel_shuffle_module(x_transposed_BCHW)
            # 对张量进行变换，B C H W→ B H W C
            x_shuffled_BHWC = x_shuffled.permute(0, 2, 3, 1).contiguous()
            # 一致性
            x = x_shuffled_BHWC

        x_original = x

        # 提取出 SS2D多头数 的切片，存储在x_chunks列表中
        x_chunks = torch.chunk(x, self.head, dim=3)

        # 处理第一个切片，存储在级联处理结果列表x_cascaded中
        # 处理公式：一支路进入SS2D，二支路进行 规模因子 的放大后与一支路残差连接
        x_cascaded = [self.VSSBlock(x_chunks[0]) + self.VSSBlock_skip_scale * x_chunks[0]]
        # 循环处理剩余切片
        for i in range(1, self.head):
            # 前序相加。级联处理结果列表x_cascaded最后一个元素 与 当前迭代的x_chunks元素 进行相加
            x_prev_sum = x_cascaded[-1] + x_chunks[i]
            # 处理 前序相加后的 切片
            # 处理公式：一支路进入SS2D，二支路进行 规模因子 的放大后与scale一支路残差连接
            x_cascaded.append(self.VSSBlock(x_prev_sum) + self.VSSBlock_skip_scale * x_prev_sum)

        # 将 级联处理结果列表x_cascaded 的全部切片 按照C维度拼接
        x_cat = torch.cat(x_cascaded, dim=3)
        # 进行残差链接
        x_residual = self.CVM_skip_scale * x_original + x_cat
        # 进行层归一化
        x_LN = self.layerNorm(x_residual)
        return x_LN


class SpectralFeatureFusionBlock(torch.nn.Module):
    def __init__(self, input_channels, time_domain_size, reduction_ratio=8):
        super().__init__()
        # 第一部分 传入参数初始化
        # 定义 时域x大小（W）
        self.time_domain_size_x = time_domain_size
        # 定义 时域y大小（H）
        self.time_domain_size_y = time_domain_size

        # 第二部分 模块初始化
        # 定义 可学习参数
        self.frequency_coefficient = torch.nn.Parameter(torch.rand(4))
        # 计算 预计算矩阵
        self.register_buffer('precomputed_matrix',
                             self.construct_all_frequency_matrix(self.time_domain_size_x, self.time_domain_size_y, 8,
                                                                 8))
        # 定义 多层感知机
        self.mlp_ca = torch.nn.Sequential(
            torch.nn.Linear(input_channels, input_channels // reduction_ratio, bias=True),  # 从 c -> c/r
            torch.nn.ReLU(),
            torch.nn.Linear(input_channels // reduction_ratio, input_channels, bias=True),  # 从 c/r -> c
            torch.nn.Sigmoid()
        )
        # 定义 层归一化
        self.norm = torch.nn.BatchNorm2d(input_channels)
        # 定义 输出激活函数
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        # 指示输入x的形状格式，并获取x的形状大小
        B, C, H, W = x.shape
        x_original = x

        # x与预计算矩阵相乘，得到各个通道的离散余弦变换频谱图
        x_dct = torch.einsum('b c h w, v u h w -> b c v u', x, self.precomputed_matrix)
        # 计算 各个频率组分量的平均值，并加权求和，输出形状为(B, C)
        x_BC = self.construct_2DDCT_frequency_feature_value(B, C, x_dct)
        # 进入多层感知机学习，结果变形为(B, C, 1, 1)
        x_BC11 = self.mlp_ca(x_BC).view(B, C, 1, 1)
        # x 与 学习后的权值 相乘
        x_CA = x * x_BC11.expand_as(x)
        # 残差连接
        x_residual = x_original + x_CA

        # x_BHWC = x_residual.permute(0, 2, 3, 1).contiguous()
        # 进行 批量归一化
        x_n = self.norm(x_residual)
        # x_BCHW = x_n.permute(0, 3, 1, 2).contiguous()
        # 激活函数非线化后输出
        x_sigmoid = self.act(x_n)

        return x_sigmoid

    def construct_2DDCT_frequency_feature_value(self, batch, channel, input_BCVU):
        # 取出 各个通道的 频率最低点
        lowest_point = input_BCVU[:, :, 0, 0]
        # 计算 各个通道的 左上1/16的频率 的幅值总和
        low_area = input_BCVU[:, :, 0:2, 0:2].sum(dim=-1).sum(dim=-1)
        # 计算 各个通道的 左上1/4的频率 的幅值总和
        medium_area = input_BCVU[:, :, 0:4, 0:4].sum(dim=-1).sum(dim=-1)
        # 计算 各个通道的 全部频率的 幅值总和
        high_area = input_BCVU[:, :, 0:8, 0:8].sum(dim=-1).sum(dim=-1)

        # 计算 低频区 的幅值平均值
        low_area_avg = (low_area - lowest_point) / 3
        # 计算 中频区 的幅值平均值
        medium_area_avg = (medium_area - low_area) / 12
        # 计算 高频区 的幅值平均值
        high_area_avg = (high_area - medium_area) / 48

        # 使用 可学习参数 对 各个频率区的幅值平均值 进行加权求和
        BC = lowest_point * self.frequency_coefficient[0] + low_area_avg * self.frequency_coefficient[1] + \
             medium_area_avg * self.frequency_coefficient[2] + high_area_avg * self.frequency_coefficient[3]

        # 变形为BC
        output_BC = BC.view(batch, channel)
        return output_BC

    def DCT2D_basis_formula_with_normalization(self, time_domain_position_x, time_domain_position_y,
                                               time_domain_size_x, time_domain_size_y,
                                               frequency_domain_position_u, frequency_domain_position_v):
        # 生成单一频率 单一时域位置的 余弦基
        # 计算 2D余弦基
        basis = np.cos(
            ((2 * time_domain_position_x + 1) * frequency_domain_position_u * np.pi) / (2 * time_domain_size_x)) * \
                np.cos(
                    ((2 * time_domain_position_y + 1) * frequency_domain_position_v * np.pi) / (2 * time_domain_size_y))

        # 计算 u的归一化系数
        if frequency_domain_position_u == 0:
            normalization_coefficient_u = 1 / np.sqrt(time_domain_size_x)
        else:
            normalization_coefficient_u = np.sqrt(2) / np.sqrt(time_domain_size_x)

        # 计算 v的归一化系数
        if frequency_domain_position_v == 0:
            normalization_coefficient_v = 1 / np.sqrt(time_domain_size_y)
        else:
            normalization_coefficient_v = np.sqrt(2) / np.sqrt(time_domain_size_y)

        # 归一化的2D余弦基为 2D余弦基 乘以 u的归一化系数 乘以 v的归一化系数
        normalized_basis = basis * normalization_coefficient_u * normalization_coefficient_v
        return normalized_basis

    # 计算 指定频率 全部时域位置 的 归一化余弦基矩阵
    def construct_single_frequency_matrix(self, time_domain_size_x, time_domain_size_y, frequency_domain_position_u,
                                          frequency_domain_position_v):
        # 定义 单一频率 全部时域位置 的归一化余弦基矩阵
        single_frequency_matrix = torch.zeros(time_domain_size_x, time_domain_size_y)

        for time_domain_position_y in range(0, time_domain_size_y):
            for time_domain_position_x in range(0, time_domain_size_x):
                # 计算 指定频率的 全部时域位置的 归一化余弦基
                single_frequency_matrix[time_domain_position_y, time_domain_position_x] = \
                    self.DCT2D_basis_formula_with_normalization(time_domain_position_x, time_domain_position_y,
                                                                time_domain_size_x, time_domain_size_y,
                                                                frequency_domain_position_u,
                                                                frequency_domain_position_v)

        return single_frequency_matrix

    # 计算 全部频率 全部时域位置 的 归一化余弦基矩阵
    def construct_all_frequency_matrix(self, time_domain_size_x, time_domain_size_y, frequency_domain_size_u,
                                       frequency_domain_size_v):
        # 定义 全部频率 全部时域位置 的归一化余弦基矩阵
        # 形状为
        all_frequency_matrix = torch.zeros(frequency_domain_size_u, frequency_domain_size_v, time_domain_size_x,
                                           time_domain_size_y)

        for frequency_domain_position_v in range(0, frequency_domain_size_v):
            for frequency_domain_position_u in range(0, frequency_domain_size_u):
                # 计算 本次迭代频率的 全部时域位置的 归一化余弦基
                single_frequency_matrix = self.construct_single_frequency_matrix(time_domain_size_x, time_domain_size_y,
                                                                                 frequency_domain_position_u,
                                                                                 frequency_domain_position_v)

                # 将 本次迭代频率的 全部时域位置的 归一化余弦基 加入到 对应本次迭代频率对应 频域位置
                all_frequency_matrix[frequency_domain_position_v, frequency_domain_position_u, :,
                :] = single_frequency_matrix

        return all_frequency_matrix


class FeatureEnhanceBlock(torch.nn.Module):
    def __init__(self, input_channels, input_size, sff_reduction_ratio=8):
        super().__init__()
        # 定义 SFF
        self.SFF = SpectralFeatureFusionBlock(input_channels=input_channels, time_domain_size=input_size,
                                              reduction_ratio=sff_reduction_ratio)

        # 定义 深度卷积
        self.DWconv = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=5, padding=2,
                                groups=input_channels)
        self.DWconv2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=5, padding=2,
                                 groups=input_channels)

        # 定义 Relu激活函数
        self.ReLU = nn.ReLU()

    def forward(self, x):
        # 指示输入x的形状格式，并获取x的形状大小
        B, C, H, W = x.shape
        # 记录原始x
        x_original = x

        # 进入DWConv
        x_DWConv0 = self.ReLU(self.DWconv(x))
        x_DWConv1 = self.ReLU(self.DWconv2(x_DWConv0))
        # 残差链接
        x_residual = x_DWConv1 + x_original
        # 进入SFF
        x_SFF = self.SFF(x_residual)

        return x_SFF


class CVMFEBlock(nn.Module):
    def __init__(self, input_channel, input_size, cvm_head, cvm_hidden_state, cvm_ss2d_expand_ratio, cvm_drop_path_rate,
                 cvm_channel_shuffle, cvm_depth,
                 fe_sff_reduction_ratio, fe_depth):
        super(CVMFEBlock, self).__init__()
        # 第一部分：参数定义
        # 定义 输入特征图通道数量
        self.input_channel = input_channel
        # 定义 输入特征图尺寸大小
        self.input_size = input_size

        # ——————CVM参数——————
        # 定义 CVMFEBlock中 CascadedVMambaBlock的 多头数量
        self.cvm_head = cvm_head
        # 定义 CVMFEBlock中 CascadedVMambaBlock的 隐状态维度量
        self.cvm_hidden_state = cvm_hidden_state
        # 定义 CVMFEBlock中 CascadedVMambaBlock的 ss2d内部数据扩张比率
        self.cvm_ss2d_expand_ratio = cvm_ss2d_expand_ratio
        # 定义 CVMFEBlock中 CascadedVMambaBlock的 整体随机失活率
        self.cvm_drop_path_rate = cvm_drop_path_rate
        # 定义 CVMFEBlock中 CascadedVMambaBlock的 是否使用通道乱序
        self.cvm_channel_shuffle = cvm_channel_shuffle
        # 定义 CVMFEBlock中 CascadedVMambaBlock的 级联数量
        self.cvm_depth = cvm_depth

        # ——————FE参数——————
        # 定义 CVMFEBlock-FeatureEnhanceBlock-SpectralFeatureFusionBlock的 MLP瓶颈系数
        self.fe_sff_reduction_ratio = fe_sff_reduction_ratio
        # 定义 CVMFEBlock-FeatureEnhanceBlock 级联数量
        self.fe_depth = fe_depth

        # 第二部分：模块初始化
        # 批量定义 CascadedVMambaBlock
        for i in range(0, cvm_depth):
            CVM = CascadedVMambaBlock(input_channel=self.input_channel,
                                      head=self.cvm_head,
                                      hidden_state=self.cvm_hidden_state,
                                      ss2d_expand_ratio=self.cvm_ss2d_expand_ratio,
                                      drop_path_rate=self.cvm_drop_path_rate,
                                      channel_shuffle=self.cvm_channel_shuffle)
            setattr(self, 'CVM%d' % i, CVM)

        # 批量定义 前置FeatureEnhanceBlock
        for i in range(0, fe_depth):
            FEA = FeatureEnhanceBlock(input_channels=self.input_channel,
                                      input_size=self.input_size,
                                      sff_reduction_ratio=fe_sff_reduction_ratio, )
            setattr(self, 'FEA%d' % i, FEA)

        # 批量定义 后置FeatureEnhanceBlock
        for i in range(0, fe_depth):
            FEB = FeatureEnhanceBlock(input_channels=self.input_channel,
                                      input_size=self.input_size,
                                      sff_reduction_ratio=fe_sff_reduction_ratio, )
            setattr(self, 'FEB%d' % i, FEB)

    def forward(self, x):
        # 指示输入x的形状格式，并获取x的形状大小
        B, H, W, C = x.shape

        # 第一次FE
        # 对张量进行变换，B H W C → B C H W
        x = x.permute(0, 3, 1, 2).contiguous()
        # 记录经过FEA前的原始x
        x_original_FE1 = x
        for i in range(0, self.fe_depth):
            FEA = getattr(self, 'FEA%d' % i)
            x = FEA(x)
        # 若 FE的级联数量不止一个
        # if self.fe_depth != 1:
        # x = x + x_original_FE1
        # 对张量进行变换，B C H W→ B H W C
        x = x.permute(0, 2, 3, 1).contiguous()

        # 记录经过CVM前的原始x
        x_original_cvm = x
        for i in range(0, self.cvm_depth):
            CVM = getattr(self, 'CVM%d' % i)
            x = CVM(x)
        # 若 CVM的级联数量不止一个
        # if self.cvm_depth != 1:
        # x = x + x_original_cvm

        # 对张量进行变换，B H W C → B C H W
        x = x.permute(0, 3, 1, 2).contiguous()
        # 记录经过FEB前的原始x
        x_original_FE2 = x
        for i in range(0, self.fe_depth):
            FEB = getattr(self, 'FEB%d' % i)
            x = FEB(x)
        # 若 FE的级联数量不止一个
        # if self.fe_depth != 1:
        # x = x + x_original_FE2

        # 对张量进行变换，B C H W→ B H W C
        x_BHWC = x.permute(0, 2, 3, 1).contiguous()
        return x_BHWC


class CVMFELayer(nn.Module):
    def __init__(self, depth, input_channel, input_size, cvm_head, cvm_hidden_state, cvm_ss2d_expand_ratio,
                 cvm_drop_path_rate, cvm_channel_shuffle, cvm_depth,
                 fe_sff_reduction_ratio, fe_depth):
        super(CVMFELayer, self).__init__()
        # 定义 CVMFEBlock的级联数量
        self.depth = depth
        # 批量定义 CVMFEBlock
        for i in range(0, self.depth):
            CVMFEBlock_ = CVMFEBlock(input_channel, input_size, cvm_head, cvm_hidden_state, cvm_ss2d_expand_ratio,
                                     cvm_drop_path_rate, cvm_channel_shuffle, cvm_depth,
                                     fe_sff_reduction_ratio, fe_depth)
            setattr(self, 'CVMFEBlock_%d' % i, CVMFEBlock_)

    def forward(self, x):
        for i in range(0, self.depth):
            CVMFEBlock_ = getattr(self, 'CVMFEBlock_%d' % i)
            x = CVMFEBlock_(x)
        return x


class PyramidPathFeatureFusionBlock(nn.Module):
    def __init__(self, num_path, ):
        super().__init__()
        self.a = 1


class CrossPathFeatureFusionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = 1


class ResidualBlock(nn.Module):
    def __init__(self, input_channel, input_size, rb_drop_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.DWconv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=3, padding=1,
                                groups=input_channel)
        self.PWconv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1)
        self.act = nn.ReLU()
        self.SFF = SpectralFeatureFusionBlock(input_channels=input_channel, time_domain_size=input_size)
        self.norm = nn.BatchNorm2d(input_channel)
        self.drop = nn.Dropout(rb_drop_rate)

        # 对DWconv和PWconv进行Kaiming初始化
        nn.init.kaiming_normal_(self.DWconv.weight, a=0, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.PWconv.weight, a=0, mode='fan_out', nonlinearity='relu')
        if self.DWconv.bias is not None:
            nn.init.constant_(self.DWconv.bias, 0)
        if self.PWconv.bias is not None:
            nn.init.constant_(self.PWconv.bias, 0)

    def forward(self, x):
        B, H, W, C = x.shape
        x_BCHW = x.permute(0, 3, 1, 2).contiguous()
        x_original = x_BCHW
        x_DWconv = self.act(self.DWconv(x_BCHW))
        x_PWconv = self.act(self.PWconv(x_DWconv))
        x_drop = self.drop(x_PWconv)
        x_residual = x_drop + x_original
        # x_norm = self.norm(x_residual)
        x_sff = self.SFF(x_residual)
        X_BHWC = x_sff.permute(0, 2, 3, 1).contiguous()
        return X_BHWC


class CVM_UNet(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 input_size=224,
                 patch_size=4,
                 embedding_ratio=2,
                 encoder_cvmfeblock_cvm_heads_list=[2, 2, 4, 4],
                 encoder_cvmfeblock_cvm_depth_list=[2, 2, 1, 2],
                 encoder_cvmfeblock_depth_list=[2, 2, 2, 2],
                 decoder_cvmfeblock_cvm_heads_list=[4, 4, 2, 2],
                 decoder_cvmfeblock_cvm_depth_list=[2, 1, 2, 2],
                 decoder_cvmfeblock_depth_list=[2, 2, 2, 2],
                 cvm_fe_ratio=3,
                 cvm_ss2d_expand_ratio=2,
                 cvm_channel_shuffle=False,
                 cvm_drop_path_rate=0.2,
                 fe_single=True,
                 rb_drop_path_rate=0.2,
                 ):
        super().__init__()
        # 第一部分：参数定义
        # ——————全局参数——————
        # 定义 输入图片的通道数
        self.input_channels = input_channels
        # 定义 语义分割的类别数
        self.num_classes = num_classes
        # 定义 输入图片的大小
        self.input_size = input_size

        # ——————PatchEmbedding参数——————
        # 定义 PatchEmbedding的Patch大小
        self.patch_size = patch_size
        # 定义 PatchEmbedding的升维比率
        self.embedding_ratio = embedding_ratio
        # 若 升维比率 不是大于等于1，抛出assert异常
        assert embedding_ratio >= 1, 'embedding_ratio must be greater than or equal to 1'
        # 自动计算 第一层Encoder输入通道数
        # 计算公式：输入图片的总像素量(self.input_size * self.input_size * self.input_channels) 除以 patch数量(self.input_size // self.patch_size)^2 乘以 升维比率(self.embedding_ratio)
        self.encoder0_input_channels = int(((self.input_size * self.input_size * self.input_channels) / (
                self.input_size / self.patch_size) / (self.input_size / self.patch_size)) * self.embedding_ratio)

        # ——————其他参数——————
        # 定义 各层Encoder&Decoder的 CVMFELayer中的 CVMFEBlock中的 CascadedVMambaBlock的 随机失活率
        self.CVM_drop_path_rate = cvm_drop_path_rate
        # self.sff_drop_path_rate = sff_drop_path_rate
        # 定义 Skip Connection中的 ResidualBlock的 随机失活率
        self.RB_drop_path_rate = rb_drop_path_rate
        # 定义 CascadedVMambaBlock与FeatureEnhanceBlock 在全部CVMFEBlock中的 比率
        self.CVM_FE_radio = cvm_fe_ratio
        # 定义 全部CVMFEBlock中的 CascadedVMambaBlock中的 SS2D内部数据扩张比率
        self.SS2D_expand_ratio = cvm_ss2d_expand_ratio
        # 定义 全部CVMFEBlock中的 CascadedVMambaBlock中的 是否使用通道乱序
        self.CVM_channel_shuffle = cvm_channel_shuffle

        # ——————Encoder参数——————
        # 定义 各层Encoder的 CVMFELayer中的 CVMFEBlock中的 CascadedVMambaBlock的 多头数量
        self.encoder_CVMFEBlock_CVM_heads_list = encoder_cvmfeblock_cvm_heads_list
        # 定义 各层Encoder的 CVMFELayer中的 CVMFEBlock中的 CascadedVMambaBlock的 级联数量列表
        self.encoder_CVMFEBlock_CVM_depth_list = encoder_cvmfeblock_cvm_depth_list
        # 定义 各层Encoder的 CVMFELayer中的 CVMFEBlock的 级联数量
        self.encoder_CVMFEBlock_depth_list = encoder_cvmfeblock_depth_list
        # 计算 各层Encoder的 输入特征图的大小
        self.encoder_input_size_list = list(
            (self.input_size // self.patch_size) // (pow(2, i)) for i in
            range(0, len(self.encoder_CVMFEBlock_depth_list)))
        # 自动计算 各层Encoder中输入通道数列表
        # 计算公式：第0层Encoder输入通道数(self.encoder0_input_channels) 乘以 2^当前encoder层级(以0起算)
        self.encoder_input_channels_list = list(
            pow(2, i) * self.encoder0_input_channels for i in range(0, len(self.encoder_CVMFEBlock_CVM_depth_list)))

        # 若 各层Encoder&Decoder的 ”CVMFELayer中的 CVMFEBlock中的 FeatureEnhanceBlock的 级联数量为1的 标志位“ 挂起
        if fe_single:
            # 自动计算 各层Encoder的 CVMFELayer中的 CVMFEBlock中的 FeatureEnhanceBlock的级联数量列表
            self.encoder_CVMFEBlock_FE_depth_list = [1] * len(self.encoder_CVMFEBlock_CVM_depth_list)
        else:
            # 自动计算 各层Encoder的 CVMFELayer中的 CVMFEBlock中的 FeatureEnhanceBlock的级联数量列表
            # 计算公式：各层Encoder的 CVMFELayer中的 CVMFEBlock中的 CascadedVMambaBlock的 级联数量列表（self.encoder_CVMFEBlock_CVM_depth_list）的各元素 乘以 cvm_fe_ratio
            self.encoder_CVMFEBlock_FE_depth_list = list(
                self.CVM_FE_radio * cvm_depth for cvm_depth in self.encoder_CVMFEBlock_CVM_depth_list)

        # ——————Decoder参数——————
        # 定义 各层Decoder的 CVMFELayer中的 CVMFEBlock中的 CascadedVMambaBlock的 多头数量
        self.decoder_CVMFEBlock_CVM_heads_list = decoder_cvmfeblock_cvm_heads_list
        # 定义 各层Decoder的 CVMFELayer中的 CVMFEBlock中的 CascadedVMambaBlock的 级联数量列表
        self.decoder_CVMFEBlock_CVM_depth = decoder_cvmfeblock_cvm_depth_list
        # 定义 各层Decoder的 CVMFELayer中的 CVMFEBlock的 级联数量
        self.decoder_CVMFEBlock_depth_list = decoder_cvmfeblock_depth_list
        # 计算 各层Decoder的 输入特征图的大小
        self.decoder_input_size_list = list(
            (self.input_size // self.patch_size) // (pow(2, len(self.decoder_CVMFEBlock_depth_list) - i - 1)) for i in
            range(0, len(self.decoder_CVMFEBlock_depth_list)))
        # 自动计算 各层Decoder中输入通道数列表
        # 计算公式：第一层Encoder输入通道数(self.encoder0_input_channels) 乘以 2^(全部decoder层数 - 当前decoder层级(以底数起算) - 1)
        self.decoder_input_channels_list = list(
            pow(2, (len(self.decoder_CVMFEBlock_CVM_depth) - i - 1)) * self.encoder0_input_channels for i in
            range(0, len(self.decoder_CVMFEBlock_CVM_depth)))

        # 若 各层Encoder&Decoder的 ”CVMFELayer中的 CVMFEBlock中的 FeatureEnhanceBlock的 级联数量为1的 标志位“ 挂起
        if fe_single:
            # 自动计算 各层Decoder的 CVMFELayer中的 CVMFEBlock中的 FeatureEnhanceBlock的级联数量列表
            self.decoder_CVMFEBlock_FE_depth_list = [1] * len(self.decoder_CVMFEBlock_CVM_depth)
        else:
            # 自动计算 各层Decoder的 CVMFELayer中的 CVMFEBlock中的 FeatureEnhanceBlock的级联数量列表
            # 计算公式：各层Decoder中的 CVMFELayer中的 CVMFEBlock中的 CascadedVMambaBlock的 级联数量列表（self.decoder_CVMFEBlock_CVM_depth）的各元素 乘以 cvm_fe_ratio
            self.decoder_CVMFEBlock_FE_depth_list = list(
                self.CVM_FE_radio * cvm_depth for cvm_depth in self.decoder_CVMFEBlock_CVM_depth)

        # 第二部分：模块初始化
        # 定义 词嵌入模块
        self.patch_embedding = PatchEmbed2D(patch_size=self.patch_size, in_chans=self.input_channels,
                                            embed_dim=self.encoder0_input_channels)
        # 定义 最后一个的词扩张模块
        self.final_expanding = Final_PatchExpand2D(dim=self.decoder_input_channels_list[-1], dim_scale=self.patch_size)
        # 定义 分类逐点卷积
        self.final_conv = nn.Conv2d(in_channels=self.decoder_input_channels_list[-1] // self.patch_size,
                                    out_channels=self.num_classes, kernel_size=1)
        # 定义 sigmoid激活函数
        self.activation = torch.nn.Sigmoid()

        # 批量定义 各层Encoder的CVMFELayer与PatchMerging
        for depth in range(0, len(self.encoder_CVMFEBlock_depth_list)):
            if depth != (len(self.encoder_CVMFEBlock_depth_list) - 1):
                encoder = nn.Sequential(
                    CVMFELayer(depth=self.encoder_CVMFEBlock_depth_list[depth],
                               input_channel=self.encoder_input_channels_list[depth],
                               input_size=self.encoder_input_size_list[depth],
                               cvm_head=self.encoder_CVMFEBlock_CVM_heads_list[depth],
                               cvm_hidden_state=16,
                               cvm_ss2d_expand_ratio=self.SS2D_expand_ratio,
                               cvm_channel_shuffle=self.CVM_channel_shuffle,
                               cvm_drop_path_rate=self.CVM_drop_path_rate,
                               cvm_depth=self.encoder_CVMFEBlock_CVM_depth_list[depth],
                               fe_sff_reduction_ratio=8,
                               fe_depth=self.encoder_CVMFEBlock_FE_depth_list[depth],
                               ),
                    PatchMerging2D(dim=self.encoder_input_channels_list[depth])
                )
            else:
                encoder = CVMFELayer(depth=self.encoder_CVMFEBlock_depth_list[depth],
                                     input_channel=self.encoder_input_channels_list[depth],
                                     input_size=self.encoder_input_size_list[depth],
                                     cvm_head=self.encoder_CVMFEBlock_CVM_heads_list[depth],
                                     cvm_hidden_state=16,
                                     cvm_ss2d_expand_ratio=self.SS2D_expand_ratio,
                                     cvm_channel_shuffle=self.CVM_channel_shuffle,
                                     cvm_drop_path_rate=self.CVM_drop_path_rate,
                                     cvm_depth=self.encoder_CVMFEBlock_CVM_depth_list[depth],
                                     fe_sff_reduction_ratio=8,
                                     fe_depth=self.encoder_CVMFEBlock_FE_depth_list[depth],
                                     )
            setattr(self, 'encoder%d' % depth, encoder)

        # 批量定义 各层Decoder的CVMFELayer与PatchMerging
        for depth in range(0, len(self.decoder_CVMFEBlock_depth_list)):
            if depth != 0:
                decoder = nn.Sequential(
                    PatchExpand2D(dim=self.decoder_input_channels_list[depth]),
                    CVMFELayer(depth=self.decoder_CVMFEBlock_depth_list[depth],
                               input_channel=self.decoder_input_channels_list[depth],
                               input_size=self.decoder_input_size_list[depth],
                               cvm_head=self.decoder_CVMFEBlock_CVM_heads_list[depth],
                               cvm_hidden_state=16,
                               cvm_ss2d_expand_ratio=self.SS2D_expand_ratio,
                               cvm_channel_shuffle=self.CVM_channel_shuffle,
                               cvm_drop_path_rate=self.CVM_drop_path_rate,
                               cvm_depth=self.decoder_CVMFEBlock_CVM_depth[depth],
                               fe_sff_reduction_ratio=8,
                               fe_depth=self.decoder_CVMFEBlock_FE_depth_list[depth],
                               )
                )
            else:
                decoder = CVMFELayer(depth=self.decoder_CVMFEBlock_depth_list[depth],
                                     input_channel=self.decoder_input_channels_list[depth],
                                     input_size=self.decoder_input_size_list[depth],
                                     cvm_head=self.decoder_CVMFEBlock_CVM_heads_list[depth],
                                     cvm_hidden_state=16,
                                     cvm_ss2d_expand_ratio=self.SS2D_expand_ratio,
                                     cvm_channel_shuffle=self.CVM_channel_shuffle,
                                     cvm_drop_path_rate=self.CVM_drop_path_rate,
                                     cvm_depth=self.decoder_CVMFEBlock_CVM_depth[depth],
                                     fe_sff_reduction_ratio=8,
                                     fe_depth=self.decoder_CVMFEBlock_FE_depth_list[depth])
            setattr(self, 'decoder%d' % depth, decoder)

        # 批量定义 SkipConnection中的 ResidualBlock
        for depth in range(0, len(self.encoder_CVMFEBlock_depth_list)):
            residual_block = ResidualBlock(input_channel=self.encoder_input_channels_list[depth],
                                           input_size=self.encoder_input_size_list[depth],
                                           rb_drop_rate=self.RB_drop_path_rate
                                           )
            setattr(self, 'residual_block%d' % depth, residual_block)

    def forward(self, x):
        encoder_skip_connection = []
        x = self.patch_embedding(x)
        encoder_skip_connection.append(x)

        for depth in range(0, len(self.encoder_CVMFEBlock_depth_list)):
            encoder = getattr(self, 'encoder%d' % depth)
            x = encoder(x)
            if depth != (len(self.encoder_CVMFEBlock_depth_list) - 1):
                encoder_skip_connection.append(x)

        for depth in range(0, len(self.encoder_CVMFEBlock_depth_list)):
            residual_block = getattr(self, 'residual_block%d' % depth)
            # 对张量进行变换，B H W C → B C H W x = x.permute(0, 3, 1, 2).contiguous()
            encoder_skip_connection[depth] = residual_block(encoder_skip_connection[depth])

        encoder_skip_connection.reverse()
        for depth in range(0, len(self.decoder_CVMFEBlock_depth_list)):
            decoder = getattr(self, 'decoder%d' % depth)
            if depth != 0:
                x = x + encoder_skip_connection[depth - 1]
            x = decoder(x)

        x = x + encoder_skip_connection[-1]
        x_fin = self.final_expanding(x)
        x_per = x_fin.permute(0, 3, 1, 2)
        x_class = self.final_conv(x_per)

        if self.num_classes == 1:
            return self.activation(x_class)
        else:
            return x_class


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data = torch.rand([8, 3, 224, 224]).float().to(device)
    model = CVM_UNet(input_size=224).to(device)
    # out = model(data)
    calculating_params_flops(model, 224)
    # print('out.shape:', out.shape)
