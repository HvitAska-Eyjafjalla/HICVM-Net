from configs.config_universal import UniversalConfig

class CVM_UNetConfig:
    def __init__(self, UniversalConfig):
        # 输入通道数 - 已根据 通用配置文件UniversalConfig 进行自动配置
        self.input_channels = UniversalConfig.input_channels
        # 语义分割类型数 - 已根据 通用配置文件UniversalConfig 进行自动配置
        self.num_classes = UniversalConfig.num_classes
        # 输入图片大小 - 已根据 通用配置文件UniversalConfig 进行自动配置
        self.input_size = UniversalConfig.input_size_h
        # PatchEmbedding的Patch大小
        self.patch_size = 4
        # PatchEmbedding的升维比率
        self.embedding_ratio = 2
        # 各层Encoder的 CVMFELayer中的 CVMFEBlock中的 CascadedVMambaBlock的 多头数量
        self.encoder_cvmfeblock_cvm_heads_list = [2, 2, 4, 4]
        # 各层Encoder的 CVMFELayer中的 CVMFEBlock中的 CascadedVMambaBlock的 级联数量列表
        self.encoder_cvmfeblock_cvm_depth_list = [2, 2, 1, 2]
        # 各层Encoder的 CVMFELayer中的 CVMFEBlock的 级联数量
        self.encoder_cvmfeblock_depth_list = [2, 2, 2, 2]
        # 各层Decoder的 CVMFELayer中的 CVMFEBlock中的 CascadedVMamba1Block的 多头数量
        self.decoder_cvmfeblock_cvm_heads_list = [4, 4, 2, 2]
        # 各层Decoder的 CVMFELayer中的 CVMFEBlock中的 CascadedVMambaBlock的 级联数量列表
        self.decoder_cvmfeblock_cvm_depth_list = [2, 1, 2, 2]
        # 各层Decoder的 CVMFELayer中的 CVMFEBlock的 级联数量
        self.decoder_cvmfeblock_depth_list = [2, 2, 2, 2]
        # CascadedVMambaBlock与FeatureEnhanceBlock 在全部CVMFEBlock中的 比率
        self.cvm_fe_ratio = 2
        # 全部CVMFEBlock中的 CascadedVMambaBlock中的 SS2D内部数据扩张比率
        self.cvm_ss2d_expand_ratio = 2
        # 全部CVMFEBlock中的 CascadedVMambaBlock中的 是否使用通道乱序
        self.cvm_channel_shuffle = False
        # 各层Encoder&Decoder的 CVMFELayer中的 CVMFEBlock中的 CascadedVMambaBlock的 随机失活率
        self.cvm_drop_path_rate = 0.2
        # 全部CVMFELayer中的 CVMFEBlock中的 FeatureEnhanceBlock的 级联数量为1的 标志位
        self.fe_single = True
        # Skip Connection中的 ResidualBlock的 随机失活率
        self.rb_drop_path_rate = 0.2


class SwinUMambaConfig:
    def __init__(self, UniversalConfig):
        # 预训练路径
        self.ckpt = UniversalConfig.project_directory / 'pretrain' / UniversalConfig.execute_model / 'vssmtiny_dp01_ckpt_epoch_292.pth'
        # 输入通道数 - 已根据 通用配置文件UniversalConfig 进行自动配置
        self.num_input_channels = UniversalConfig.input_channels
        # 语义分割类型数 - 已根据 通用配置文件UniversalConfig 进行自动配置
        self.num_classes = UniversalConfig.num_classes
        # 是否使用深度监督
        self.deep_supervision = False
        # 是否使用预训练模型
        self.use_pretrain = False


class SwinPAConfig:
    def __init__(self, UniversalConfig):
        self.out_planes = UniversalConfig.num_classes