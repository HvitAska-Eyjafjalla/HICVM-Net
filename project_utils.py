import logging
import logging.handlers

import os
import random
import numpy as np
import pandas as pd
import torch
from nvidia import cudnn

from thop import profile
# from torchstat import stat
import torch.nn.functional as F

import torch.nn as nn
import math

# from torch.optim.lr_scheduler import LRScheduler
from collections import deque


class MyLogger:
    def __init__(self, logger_name, log_directory, universal_config, model_config):
        self.logger_name = logger_name
        self.log_directory = log_directory
        self.universal_config = universal_config
        self.model_config = model_config

        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.INFO)

    def creat_info_file(self):
        # 定义 日志文件名称
        info_name = self.log_directory / f'{self.logger_name}.info.log'
        # 定义 日志处理器
        info_handler = logging.handlers.TimedRotatingFileHandler(str(info_name), when='D', encoding='utf-8')
        info_handler.setLevel(logging.INFO)

        # 定义 日志前缀格式 并应用
        formatter = logging.Formatter('%(asctime)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        info_handler.setFormatter(formatter)

        # logger关联到info_handler
        self.logger.addHandler(info_handler)

    def log_UniversalConfig_info(self):
        config_dict = self.universal_config.__dict__
        log_info = f'#----------Universal Config info----------#'
        self.logger.info(log_info)
        for k, v in config_dict.items():
            if k[0] == '_':
                continue
            else:
                log_info = f'{k}: {v},'
                self.logger.info(log_info)

    def log_ModelConfig_info(self):
        config_dict = self.model_config.__dict__
        log_info = f'#----------Model Config info----------#'
        self.logger.info(log_info)
        for k, v in config_dict.items():
            if k[0] == '_':
                continue
            else:
                log_info = f'{k}: {v},'
                self.logger.info(log_info)

    def log_and_print_custom_info(self, info, indent=False):
        self.logger.info(info)
        if indent:
            print('\t'+info)
        else:
            print(info)



def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = True
    cudnn.deterministic = True


def calculating_params_flops(model, size, Mylogger):
    input = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    # 打印计算量
    print("\tFLOPs: %.4fG" % (flops / 1e9))
    # 打印参数量
    print("\tParams: %.4fM" % (params / 1e6))

    total = sum(p.numel() for p in model.parameters())
    # 打印总参数量
    print("\tTotal params: %.4fM" % (total / 1e6))

    Mylogger.logger.info('#----------Model info----------#')
    Mylogger.logger.info(f'Flops: {flops / 1e9:.4f}G, Params: {params / 1e6:.4f}M, Total params: : {total / 1e6:.4f}M')


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss


class WeightedBCE(nn.Module):

    def __init__(self, weights=[0.4, 0.6], n_labels=1):
        super(WeightedBCE, self).__init__()
        self.weights = weights
        self.n_labels = n_labels

    def forward(self, logit_pixel, truth_pixel):
        # print("====",logit_pixel.size())
        if self.n_labels == 1:
            logit = logit_pixel.view(-1).float()
            truth = truth_pixel.view(-1)
            assert (logit.shape == truth.shape)
            loss = F.binary_cross_entropy(logit, truth, reduction='none')
            # loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
            pos = (truth > 0.5).float()
            neg = (truth < 0.5).float()

            pos_weight = pos.sum().item() + 1e-12
            neg_weight = neg.sum().item() + 1e-12
            loss = (self.weights[0] * pos * loss / pos_weight + self.weights[1] * neg * loss / neg_weight).sum()

            return loss


class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5], n_labels=1):  # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights
        self.n_labels = n_labels

    def forward(self, logit, truth, smooth=1e-5):
        if (self.n_labels == 1):
            batch_size = len(logit)
            logit = logit.reshape(batch_size, -1)
            truth = truth.reshape(batch_size, -1)
            assert (logit.shape == truth.shape)

            # logit = torch.sigmoid(logit)

            p = logit.reshape(batch_size, -1)
            t = truth.reshape(batch_size, -1)
            w = truth.detach()
            w = w * (self.weights[1] - self.weights[0]) + self.weights[0]
            # p = w*(p*2-1)  #convert to [0,1] --> [-1, 1]
            # t = w*(t*2-1)
            p = w * p
            t = w * t
            intersection = (p * t).sum(-1)
            union = (p * p).sum(-1) + (t * t).sum(-1)
            dice = 1 - (2 * intersection + smooth) / (union + smooth)
            # print "------",dice.data

            loss = dice.mean()
            return loss


class WeightedDiceBCE(nn.Module):
    def __init__(self, dice_weight=1, BCE_weight=1, n_labels=1):
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5], n_labels=n_labels)
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5], n_labels=n_labels)
        self.n_labels = n_labels
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        inputs[inputs >= 0.5] = 1
        inputs[inputs < 0.5] = 0
        # print("2",np.sum(tmp))
        targets[targets > 0] = 1
        targets[targets <= 0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets):
        # inputs = inputs.contiguous().view(-1)
        # targets = targets.contiguous().view(-1)
        # print "dice_loss", self.dice_loss(inputs, targets)
        # print "focal_loss", self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        # print "dice",dice
        # print "focal",focal
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE

        return dice_BCE_loss


def get_optimizer(config, model):
    assert config.optimizer in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                                'SGD'], 'Unsupported optimizer!'

    if config.optimizer == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr=config.lr,
            rho=config.rho,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr=config.lr,
            lr_decay=config.lr_decay,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.optimizer == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.optimizer == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr=config.lr,
            lambd=config.lambd,
            alpha=config.alpha,
            t0=config.t0,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            alpha=config.alpha,
            eps=config.eps,
            centered=config.centered,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr=config.lr,
            etas=config.etas,
            step_sizes=config.step_sizes,
        )
    elif config.optimizer == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            dampening=config.dampening,
            nesterov=config.nesterov
        )
    else:  # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.05,
        )


def get_scheduler(config, optimizer):
    assert config.scheduler in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                                'CosineAnnealingWarmRestarts', 'WP_MultiStepLR',
                                'WP_CosineLR', 'AdaptiveLinearAnnealingSoftRestarts'], 'Unsupported scheduler!'
    if config.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.milestones,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.scheduler == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            threshold_mode=config.threshold_mode,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            eps=config.eps
        )
    elif config.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.scheduler == 'WP_MultiStepLR':
        lr_func = lambda \
                epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma ** len(
            [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.scheduler == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.total_epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    elif config.scheduler == 'AdaptiveLinearAnnealingSoftRestarts':
        scheduler = AdaptiveLinearAnnealingSoftRestarts(
            optimizer,
            cycle=config.cycle,
            min_learn_rate=config.min_learn_rate,
            correlation_window_size=config.correlation_window_size,
            variance_window_size=config.variance_window_size,
            correlation_threshold=config.correlation_threshold,
            variance_threshold=config.variance_threshold,
            soft_restart=config.soft_restart
        )

    return scheduler


class AdaptiveLinearAnnealingSoftRestarts:

    def __init__(self, optimizer, cycle, min_learn_rate, correlation_window_size=5, variance_window_size=10,
                 correlation_threshold=0, variance_threshold=0.0002, soft_restart=True):
        # 第一部分 传入参数初始化
        self.optimizer = optimizer
        # 定义 一个目标衰减周期的大小
        self.cycle = cycle
        # 定义 最大学习率/初始学习率
        self.base_learn_rate = [group['lr'] for group in optimizer.param_groups][0]
        # 定义 最小学习率
        self.min_learn_rate = min_learn_rate
        # 定义 计算自相关值的窗口大小
        self.correlation_window_size = correlation_window_size
        # 若 计算自相关值的窗口大小 非大于等于3，抛出assert异常
        assert correlation_window_size >= 3, 'correlation window size should be >= 3'
        # 定义 计算方差的窗口大小
        self.variance_window_size = variance_window_size
        # 若 计算方差的窗口大小 非大于等于10，抛出assert异常
        assert variance_window_size >= 10, 'variance window size should be >= 10'
        # 定义 执行加速衰减的自相关阈值
        self.correlation_threshold = correlation_threshold
        # 定义 执行回调学习率的方差阈值
        self.variance_threshold = variance_threshold
        # 定义 回调的是否为软回调
        self.soft_restart = soft_restart

        # 第二部分 内定参数初始化
        # 定义 步进量
        self.stepped_size = 0
        # 定义 当前衰减周期
        self.current_cycle = 0
        # 定义 val loss 最小值更新计数器
        self.counter = 0
        # 定义 正常衰减的衰减率
        self.decay_rate = (self.base_learn_rate - self.min_learn_rate) / self.cycle
        # 定义 上升率
        self.rising_rate = (self.base_learn_rate - self.min_learn_rate) / self.cycle
        # 定义 上次更新的学习率
        self.last_learn_rate = self.base_learn_rate
        # 定义 计算自相关值的队列数据结构（以下简称自相关队列）
        self.correlation_queue = deque(maxlen=correlation_window_size)
        # 定义 计算方差的队列数据结构（以下简称方差队列）
        self.variance_queue = deque(maxlen=variance_window_size)
        # 定义 需要加速衰减的标志位
        self.need_accelerate = False
        # 定义 需要回调学习率的标志位
        self.need_restart = False
        # 定义 回调阶段的标志位
        self.restart_phase = False
        # 定义 到达最小学习率的标志位
        self.reach_min_learn_rate = False

    def check_correlation_window(self, val_loss):
        # 给 自相关队列 加入新的元素 本训练轮次的验证损失val_loss
        self.correlation_queue.append(val_loss)

        # 若 自相关队列 满队列
        if len(list(self.correlation_queue)) == self.correlation_window_size:
            # 计算 自相关队列 全部元素的自相关值
            correlation = pd.Series(np.array(list(self.correlation_queue))).autocorr()

            # 若 自相关值 小于 执行加速衰减的自相关阈值
            if correlation <= self.correlation_threshold:
                # 需要加速衰减的标志位 挂起
                self.need_accelerate = True

            # 净空 自相关队列
            self.correlation_queue.clear()

    def check_variance_window(self):
        # 若 方差队列 满队列
        if len(list(self.variance_queue)) == self.variance_window_size:
            # 计算 方差队列 全部元素的方差值
            variance = np.var(np.array(list(self.variance_queue)))

            # 若 方差值 小于 执行回调学习率的方差阈值
            if variance <= self.variance_threshold:
                # 需要回调学习率的标志位 挂起
                self.need_restart = True

    def min_valloss_refresh_counter(self, clear):
        if clear:
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.variance_window_size:
                self.counter = max(self.counter, self.variance_window_size)

    def get_lr(self):
        # 若 为衰减阶段
        if self.restart_phase is not True:
            # 若 未到达最小学习率
            if self.reach_min_learn_rate is not True:
                # 计算 衰减后的学习率
                learn_rate = self.last_learn_rate - self.decay_rate * self.stepped_size
                # 若 衰减后的学习率 低于 最小学习率
                if learn_rate <= self.min_learn_rate:
                    # 确保 衰减后的学习率 不低于最小学习率
                    learn_rate = max(learn_rate, self.min_learn_rate)
                    # 到达最小学习率 挂起
                    self.reach_min_learn_rate = True

            # 若 到达最小学习率
            else:
                # 学习率为 最小学习率
                learn_rate = self.min_learn_rate

        # 若 为回调阶段
        else:
            # 若 回调阶段 是软回调
            if self.soft_restart:
                # 计算 回调阶段的学习率
                learn_rate = self.last_learn_rate + self.rising_rate

                # 若 回调阶段的学习率 大于等于 最大学习率/初始学习率
                if learn_rate >= self.base_learn_rate:
                    # 确保 回调阶段的学习率 不高于 最大学习率/初始学习率
                    learn_rate = min(learn_rate, self.base_learn_rate)
                    # 清除 回调阶段的标志位，代表结束回调阶段
                    self.restart_phase = False
                    # 衰减周期 自加1
                    self.current_cycle += 1

            # 若 回调阶段 是硬回调
            else:
                # 直接 回调到 最大学习率/初始学习率
                learn_rate = self.base_learn_rate
                # 清除 回调阶段的标志位，代表结束回调阶段
                self.restart_phase = False
                # 衰减周期 自加1
                self.current_cycle += 1

        # 将计算得出的 学习率 记录为 上次更新的学习率
        self.last_learn_rate = learn_rate
        return learn_rate

    def step(self, val_loss: float = 0, min_valloss_refresh: bool = False, **kwargs):
        # 若 验证集损失val_loss 未被传入到 该学习率调度策略，抛出assert异常
        assert val_loss is not None, "val loss should be delivered into the scheduler"

        # 若为 衰减阶段
        if self.restart_phase is not True:
            # 计算 自相关值，检查是否需要加速学习率衰减
            self.check_correlation_window(val_loss)
            # 若 需要加速衰减 挂起
            if self.need_accelerate:
                # 加速衰减
                self.stepped_size = 6
                # 清除 需要加速衰减的标志位
                self.need_accelerate = False
            else:
                # 正常衰减
                self.stepped_size = 1

            # 给 方差队列 加入新的元素 本训练轮次的验证损失val_loss
            self.variance_queue.append(val_loss)
            self.min_valloss_refresh_counter(min_valloss_refresh)
            if self.counter == self.variance_window_size and self.last_learn_rate <= 0.2 * self.base_learn_rate:
                # 计算 方差，检查是否需要回调学习率
                self.check_variance_window()
                # 若 需要回调学习率 挂起
                if self.need_restart:
                    # 回调阶段 挂起
                    self.restart_phase = True
                    # 计算 上升率
                    self.rising_rate = (self.base_learn_rate - self.last_learn_rate) / 4

        # 若为 回调阶段
        else:
            # 净空 自相关队列
            self.correlation_queue.clear()
            # 净空 方差队列
            self.variance_queue.clear()
            self.counter = 0
            self.reach_min_learn_rate = False

        # 更新 optimizer 的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr()

    def state_dict(self):
        return {
            'cycle': self.cycle,
            'base_learn_rate': self.base_learn_rate,
            'min_learn_rate': self.min_learn_rate,
            'correlation_window_size': self.correlation_window_size,
            'variance_window_size': self.variance_window_size,
            'correlation_threshold': self.correlation_threshold,
            'variance_threshold': self.variance_threshold,
            'soft_restart': self.soft_restart,

            'stepped_size': self.stepped_size,
            'current_cycle': self.current_cycle,
            'counter': self.counter,
            'decay_rate': self.decay_rate,
            'last_learn_rate': self.last_learn_rate,
            'correlation_queue': list(self.correlation_queue),
            'variance_queue': list(self.variance_queue),
            'need_accelerate': self.need_accelerate,
            'need_restart': self.need_restart,
            'restart_phase': self.restart_phase,
            'reach_min_learn_rate': self.reach_min_learn_rate
        }

    def load_state_dict(self, state_dict):
        self.cycle = state_dict['cycle']
        self.base_learn_rate = state_dict['base_learn_rate']
        self.min_learn_rate = state_dict['min_learn_rate']
        self.correlation_window_size = state_dict['correlation_window_size']
        self.variance_window_size = state_dict['variance_window_size']
        self.correlation_threshold = state_dict['correlation_threshold']
        self.variance_threshold = state_dict['variance_threshold']
        self.soft_restart = state_dict['soft_restart']
        self.stepped_size = state_dict['stepped_size']
        self.current_cycle = state_dict['current_cycle']
        self.counter = state_dict['counter']
        self.decay_rate = state_dict['decay_rate']
        self.last_learn_rate = state_dict['last_learn_rate']
        self.correlation_queue = deque(state_dict['correlation_queue'], maxlen=self.correlation_window_size)
        self.variance_queue = deque(state_dict['variance_queue'], maxlen=self.variance_window_size)
        self.need_accelerate = state_dict['need_accelerate']
        self.need_restart = state_dict['need_restart']
        self.restart_phase = state_dict['restart_phase']
        self.reach_min_learn_rate = state_dict['reach_min_learn_rate']
