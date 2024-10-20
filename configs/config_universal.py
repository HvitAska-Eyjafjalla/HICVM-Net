from datetime import datetime
from pathlib import Path
from project_utils import *


# 规范定义1 ： directory指的是目录，path指的是具体文件路径
class UniversalConfig:
    def __init__(self, execute_model_index, execute_pretrain_model_index, execute_dataset_index, criterion=None, optimizer=None, scheduler=None, num_workers=None):
        # 第一部分:——————设置项目参数——————
        # 定义 支持的执行模型列表
        #                          0           1             2               3               4                    5                       6               7              8
        self.support_model_list = ['CVM_UNet', 'SwinUMamba', 'CVM_UNet_FFC', 'CVM_UNet_CFF', 'CVM_UNet_Parallel', 'CVM_UNet_Independent', 'CVM_UNet_MLP', 'CVM_UNet_SE', 'CVM_UNet_FCA']
        # 设置 项目的执行模型
        self.execute_model = self.support_model_list[execute_model_index]
        # 获取 项目所在路径project_path
        self.project_directory = Path.cwd()
        # 设置 执行时间
        self.execute_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        # 定义 支持的预训练模型列表
        #                             0     1               2                3               4
        self.support_pretrain_list = [None, 'min_loss.pth', 'max_score.pth', 'max_dice.pth', 'latest.pth']
        # 设置 执行预训练模型
        self.execute_pretrain_model = self.support_pretrain_list[execute_pretrain_model_index]
        # 若 执行预训练模型不为空
        if self.execute_pretrain_model is not None:
            # 自动设置 预训练模型路径
            self.pretrain_model_path = self.project_directory / 'pretrain' / self.execute_model / self.execute_pretrain_model
        else:
            # 自动设置 预训练模型路径为空
            self.pretrain_model_path = None

        # 第二部分:——————设置数据集及其目录路径——————
        # 定义 支持的执行模型列表
        #                            0        1            2               3           4
        self.support_dataset_list = ['Glas', 'Kvasir_SEG', 'CVC_ClinicDB', 'ISIC2018', 'BUSI']
        # 设置 执行数据集
        self.execute_dataset = self.support_dataset_list[execute_dataset_index]
        # 设置 数据集路径dataset_directory 为对应的执行数据集路径
        self.dataset_directory = self.project_directory / 'datasets' / self.execute_dataset

        # 第三部分:——————设置执行结果目录路径——————
        # 设置 执行结果文件目录名称。格式为“{执行模型}__{执行数据集}__{执行时间}”
        self.result_directory_name = self.execute_model + '__' + self.execute_dataset + '__' + self.execute_time
        # 设置 执行结果目录
        self.result_directory = self.project_directory / 'results' / self.result_directory_name
        # 设置 结果日志目录
        self.log_directory = self.result_directory / 'log'
        # 设置 结果分割输出目录
        self.output_directory = self.result_directory / 'output'
        # 设置 最好模型目录
        self.best_models_directory = self.result_directory / 'best_models'

        # 第四部分:——————设置设备训练参数——————
        # 自动设置 设备类型
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        # 出现大量问题，建议不使用amp
        # 若 NVIDIA设备 可用
        if self.device.type == 'cuda':
            # 自动设置 启用NVIDIA自动混合精度训练功能，可手动设置
            # 将同时启用混合精度前向传播上下文管理器 与 混合精度反向传播梯度放大，以利用NVIDIA Tensor Core
            self.automatic_mixed_precision = False
        # 若 NVIDIA设备 不可用
        else:
            # 自动设置 停用NVIDIA自动混合精度训练功能
            self.automatic_mixed_precision = False

        # 第五部分:——————设置训练参数——————
        # 设置 分类类别量
        self.num_classes = 1
        # 设置 输入图片高度
        self.input_size_h = 224
        # 设置 输入图片宽度
        self.input_size_w = 224
        # 设置 输入图片通道数量
        self.input_channels = 3
        # 设置 指定种子
        self.seed = 1202
        # 设置 项目的批次大小
        self.batch_size = 4
        # 设置 项目的训练轮次
        self.total_epochs = 10
        # 设置 验证阶段二值化阈值
        self.evaluate_threshold = 0.5
        # 设置 梯度裁切阈值
        self.gradient_clipping = 2
        # 设置 早停法等待轮次阈值
        self.early_stopping_patience = 140
        # 设置 dataloader 并行线程数                                                                                                                                         
        if num_workers is None:
            self.num_workers = 0
        else:
            self.num_workers = num_workers

        # 第六部分:——————设置训练打印参数——————
        # 一次训练过程中 打印一次信息的轮次间隔
        self.print_interval = 5
        # 执行并打印一次完整验证信息 的轮次间隔
        self.val_interval = 1
        # 开始打印 验证结果信息的占总训练轮次的比值
        self.result_interval = 0.05 
        # 执行并打印一次完整验证信息 的轮次间隔
        self.save_interval = 1
        # 打印 剩余时间 的轮次间隔
        self.estimate_interval = 2 * self.val_interval

        # 第七部分:——————设置损失函数,优化器,学习策略——————
        # 设置 损失函数为 二元交叉熵损失函数与Dice损失函数 的联合损失函数
        # self.criterion = BceDiceLoss(wb=0.5, wd=0.5)
        if criterion == 'BceDiceLoss':
            self.criterion = BceDiceLoss()
        elif criterion is None:
            self.criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5, n_labels=self.num_classes)

        if optimizer is None:
            self.optimizer = 'AdamW'
        else:
            self.optimizer = optimizer
        assert self.optimizer in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                                  'SGD'], 'Unsupported optimizer!'
        if self.optimizer == 'Adadelta':
            self.lr = 0.01  # 默认值: 1.0 – coefficient that scale delta before it is applied to the parameters
            self.rho = 0.9  # 默认值: 0.9 – 用于计算梯度平方的运行平均值的系数
            self.eps = 1e-6  # 默认值: 1e-6 – term added to the denominator to improve numerical stability
            self.weight_decay = 0.05  # default: 0 – weight decay (L2 penalty)
        elif self.optimizer == 'Adagrad':
            self.lr = 0.01  # default: 0.01 – learning rate
            self.lr_decay = 0  # default: 0 – learning rate decay
            self.eps = 1e-10  # default: 1e-10 – term added to the denominator to improve numerical stability
            self.weight_decay = 0.05  # default: 0 – weight decay (L2 penalty)
        elif self.optimizer == 'Adam':
            self.lr = 0.001  # default: 1e-3 – learning rate
            self.betas = (0.9,
                          0.999)  # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
            self.eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
            self.weight_decay = 0.0001  # default: 0 – weight decay (L2 penalty)
            self.amsgrad = False  # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
        elif self.optimizer == 'AdamW':
            self.lr = 1e-4  # default: 1e-3 – learning rate
            self.betas = (0.9,
                          0.999)  # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
            self.eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
            self.weight_decay = 1e-2  # default: 1e-2 – weight decay coefficient
            self.amsgrad = False  # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
        elif self.optimizer == 'Adamax':
            self.lr = 2e-3  # default: 2e-3 – learning rate
            self.betas = (0.9,
                          0.999)  # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
            self.eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
            self.weight_decay = 0  # default: 0 – weight decay (L2 penalty)
        elif self.optimizer == 'ASGD':
            self.lr = 0.01  # default: 1e-2 – learning rate
            self.lambd = 1e-4  # default: 1e-4 – decay term
            self.alpha = 0.75  # default: 0.75 – power for eta update
            self.t0 = 1e6  # default: 1e6 – point at which to start averaging
            self.weight_decay = 0  # default: 0 – weight decay
        elif self.optimizer == 'RMSprop':
            self.lr = 1e-2  # default: 1e-2 – learning rate
            self.momentum = 0  # default: 0 – momentum factor
            self.alpha = 0.99  # default: 0.99 – smoothing constant
            self.eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
            self.centered = False  # default: False – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
            self.weight_decay = 0  # default: 0 – weight decay (L2 penalty)
        elif self.optimizer == 'Rprop':
            self.lr = 1e-2  # default: 1e-2 – learning rate
            self.etas = (0.5,
                         1.2)  # default: (0.5, 1.2) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors
            self.step_sizes = (1e-6, 50)  # default: (1e-6, 50) – a pair of minimal and maximal allowed step sizes
        elif self.optimizer == 'SGD':
            self.lr = 1e-6  # – learning rate
            self.momentum = 0.9  # default: 0 – momentum factor
            self.weight_decay = 0.05  # default: 0 – weight decay (L2 penalty)
            self.dampening = 0  # default: 0 – dampening for momentum
            self.nesterov = False  # default: False – enables Nesterov momentum

        if optimizer is None:
            self.scheduler = 'CosineAnnealingLR'
        else:
            self.scheduler = scheduler
        assert self.scheduler in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                                  'CosineAnnealingWarmRestarts',
                                  'WP_MultiStepLR', 'WP_CosineLR', 'AdaptiveLinearAnnealingSoftRestarts'], 'Unsupported scheduler! '
        if self.scheduler == 'StepLR':
            self.step_size = self.total_epochs // 5  # – Period of learning rate decay.
            self.gamma = 0.5  # – Multiplicative factor of learning rate decay. Default: 0.1
            self.last_epoch = -1  # – The index of last epoch. Default: -1.
        elif self.scheduler == 'MultiStepLR':
            self.milestones = [60, 120, 150]  # – List of epoch indices. Must be increasing.
            self.gamma = 0.1  # – Multiplicative factor of learning rate decay. Default: 0.1.
            self.last_epoch = -1  # – The index of last epoch. Default: -1.
        elif self.scheduler == 'ExponentialLR':
            self.gamma = 0.99  # – Multiplicative factor of learning rate decay.
            self.last_epoch = -1  # – The index of last epoch. Default: -1.
        elif self.scheduler == 'CosineAnnealingLR':
            self.T_max = 50  # – Maximum number of iterations. Cosine function period.
            self.eta_min = 0.00001  # – Minimum learning rate. Default: 0.
            self.last_epoch = -1  # – The index of last epoch. Default: -1.
        elif self.scheduler == 'ReduceLROnPlateau':
            self.mode = 'min'  # – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
            self.factor = 0.1  # – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
            self.patience = 10  # – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
            self.threshold = 0.0001  # – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
            self.threshold_mode = 'rel'  # – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
            self.cooldown = 0  # – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
            self.min_lr = 0  # – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
            self.eps = 1e-08  # – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
        elif self.scheduler == 'CosineAnnealingWarmRestarts':
            self.T_0 = 10  # – Number of iterations for the first restart.
            self.T_mult = 1  # – A factor increases T_{i} after a restart. Default: 1.
            self.eta_min = 1e-4  # – Minimum learning rate. Default: 0.
            self.last_epoch = -1  # – The index of last epoch. Default: -1.
        elif self.scheduler == 'WP_MultiStepLR':
            self.warm_up_epochs = 10
            self.gamma = 0.1
            self.milestones = [125, 225]
        elif self.scheduler == 'WP_CosineLR':
            self.warm_up_epochs = 20

        elif self.scheduler == 'AdaptiveLinearAnnealingSoftRestarts':
            self.cycle = 120
            self.min_learn_rate = 1e-6
            self.correlation_window_size = 5
            self.variance_window_size = 12
            self.correlation_threshold = 0
            self.variance_threshold = 0.0002
            self.soft_restart = True
