from configs.config_universal import *
from project_utils import *
from torch.utils.data import DataLoader
from engine import evaluate_one_epoch
from configs.config_model import *

from multi_dataset import *

from models.CVM_UNet.CVM_UNet import CVM_UNet
from models.SwinUMamba.SwinUMamba import get_swin_umamba_from_plans


def test(universal_config, model_config):
    for directory in [universal_config.log_directory, universal_config.output_directory,
                      universal_config.best_models_directory]:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)

    test_dataset = MultiDataset(universal_config=universal_config, Mylogger=None, test=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=universal_config.num_workers,
                                 drop_last=True)

    model_config_dict = {k: v for k, v in model_config.__dict__.items() if not k.startswith('__') and not callable(v)}
    # ↓—————————————————————————————————————————————————————↓
    # 以 超参数-执行模型 加载 对应模型
    if universal_config.execute_model == 'CVM_UNet':
        model = CVM_UNet(**model_config_dict)
    elif universal_config.execute_model == 'SwinUMamba':
        model = get_swin_umamba_from_plans(**model_config_dict)
    # ↑—————————————————————————————————————————————————————↑
    # 将 对应模型 加载到 设备
    model.to(device=universal_config.device)

    # 以 超参数-criterion 配置 损失函数
    criterion = universal_config.criterion
    # 以 universal_config 为 对应模型 配置 优化器
    optimizer = get_optimizer(universal_config, model)
    # 以 universal_config 为 优化器 配置 学习策略
    scheduler = get_scheduler(universal_config, optimizer)
    # 创建 混合精度反向传播梯度放大器，由automatic_mixed_precision决定是否启用
    grad_scaler = torch.cuda.amp.GradScaler(enabled=universal_config.automatic_mixed_precision)

    if universal_config.pretrain_model_path is not None:
        # 装载预训练模型
        pretrain_model = torch.load(universal_config.pretrain_model_path,
                                    map_location=torch.device(universal_config.device))
        # 加载 预训练模型的 参数
        # model.load_state_dict(pretrain_model, strict=False)
        model.load_state_dict(pretrain_model['model_state_dict'], strict=False)
        # 加载 预训练模型的 优化器参数
        # optimizer.load_state_dict(pretrain_model['optimizer_state_dict'])
        # 加载 预训练模型的 学习策略参数
        # scheduler.load_state_dict(pretrain_model['scheduler_state_dict'])

    loss = evaluate_one_epoch(test_dataloader,
                              model,
                              criterion,
                              scheduler,
                              0,
                              None,
                              universal_config,
                              universal_config.output_directory,
                              validation=False,
                              test_data_name=None
                              )

if __name__ == '__main__':
    universal_config = UniversalConfig(execute_model_index=0, execute_pretrain_model_index=3, execute_dataset_index=3, criterion='BceDiceLoss', optimizer=None, scheduler=None, num_workers=None)
    # ↓—————————————————————————————————————————————————————↓
    if universal_config.execute_model == 'CVM_UNet':
        model_config = CVM_UNetConfig(universal_config)
    elif universal_config.execute_model == 'SwinUMamba':
        model_config = SwinUMambaConfig(universal_config)
    # ↑—————————————————————————————————————————————————————↑
    test(universal_config, model_config)
