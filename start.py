# 主函数导入
from configs.config_model import *
from project_utils import *
import torch

# Preparing Dataset导入
from multi_dataset import MultiDataset
from torch.utils.data import DataLoader

# Preparing Model导入
from models.CVM_UNet.CVM_UNet import CVM_UNet
from models.CVM_UNet.CVM_UNet_FFC import CVM_UNet_FFC
from models.CVM_UNet.CVM_UNet_CFF import CVM_UNet_CFF
from models.CVM_UNet.CVM_UNet_Parallel import CVM_UNet_Parallel
from models.CVM_UNet.CVM_UNet_Independent import CVM_UNet_Independent
from models.CVM_UNet.CVM_UNet_MLP import CVM_UNet_MLP
from models.CVM_UNet.CVM_UNet_SE import CVM_UNet_SE
from models.CVM_UNet.CVM_UNet_FCA import CVM_UNet_FCA
from models.SwinUMamba.SwinUMamba import get_swin_umamba_from_plans

# Training与Testing导入
from engine import *
import time


def execute(universal_config, model_config):
    print('\nStep1: Creating result directory')
    try:
        # 检查 结果日志目录、结果分割输出目录、最好模型目录 ,不存在则创建目录
        for directory in [universal_config.log_directory, universal_config.output_directory,
                          universal_config.best_models_directory]:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
        print('\tResult directory created!')
    except:
        print('\tCaught ERROR, result directory not created.')
        exit()

    print('Step2: Creating Logger')
    # 定义 日志对象名称
    logger_name = universal_config.execute_model + '_' + universal_config.execute_dataset
    # 初始化 全局日志对象Mylogger_instance
    Mylogger_instance = MyLogger(logger_name, universal_config.log_directory, universal_config, model_config)
    # 创建 全局日志对象Mylogger_instance对应的日志文件
    Mylogger_instance.creat_info_file()
    # 日志文件 打印 universal_config信息
    Mylogger_instance.log_UniversalConfig_info()
    # 日志文件 打印 model_config信息
    Mylogger_instance.log_ModelConfig_info()
    print('\tLogger instance created')

    print('Step3: Device Initialization')
    # 为 设备 设置 随机数种子
    set_seed(universal_config.seed)
    # 清空 设备缓存
    torch.cuda.empty_cache()

    print('Step4: Preparing Dataset')
    # 打印 日志分割信息
    Mylogger_instance.logger.info('#----------Dataset Info----------#')

    # 创建 对应数据集 的训练集
    train_dataset = MultiDataset(universal_config=universal_config, Mylogger=Mylogger_instance, train=True)
    # 创建 对应数据集 的验证集
    val_dataset = MultiDataset(universal_config=universal_config, Mylogger=Mylogger_instance, validation=True)
    # 创建 对应数据集 的测试集
    test_dataset = MultiDataset(universal_config=universal_config, Mylogger=Mylogger_instance, test=True)

    # 创建 对应数据集 的训练集 的加载器
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=universal_config.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=universal_config.num_workers)
    # 创建 对应数据集 的验证集 的加载器
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=universal_config.num_workers,
                                drop_last=True)
    # 创建 对应数据集 的测试集 的加载器
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=universal_config.num_workers,
                                 drop_last=True)
    print(f'\t Dataloaders of {universal_config.execute_dataset} are ready')

    print('Step5: Preparing Model')
    # 将model_config中的参数转换为字典形式
    model_config_dict = {k: v for k, v in model_config.__dict__.items() if not k.startswith('__') and not callable(v)}
    # ↓—————————————————————————————————————————————————————↓
    # 以 超参数-执行模型 加载 对应模型
    if universal_config.execute_model == 'CVM_UNet':
        model = CVM_UNet(**model_config_dict)
    elif universal_config.execute_model == 'SwinUMamba':
        model = get_swin_umamba_from_plans(**model_config_dict)

    elif universal_config.execute_model == 'CVM_UNet_FFC':
        model = CVM_UNet_FFC(**model_config_dict)
    elif universal_config.execute_model == 'CVM_UNet_CFF':
        model = CVM_UNet_CFF(**model_config_dict)
    elif universal_config.execute_model == 'CVM_UNet_Parallel':
        model = CVM_UNet_Parallel(**model_config_dict)
    elif universal_config.execute_model == 'CVM_UNet_Independent':
        model = CVM_UNet_Independent(**model_config_dict)
    elif universal_config.execute_model == 'CVM_UNet_SE':
        model = CVM_UNet_SE(**model_config_dict)
    elif universal_config.execute_model == 'CVM_UNet_MLP':
        model = CVM_UNet_MLP(**model_config_dict)
    elif universal_config.execute_model == 'CVM_UNet_FCA':
        model = CVM_UNet_FCA(**model_config_dict)
    # ↑—————————————————————————————————————————————————————↑
    # 若 超参数-执行模型 为非法
    else:
        raise Exception('model in not right!')

    # 将 对应模型 加载到 设备
    model.to(device=universal_config.device)
    print(f'\t{universal_config.execute_model} is ready, calculating metrics...')
    # 计算 对应模型的参数量、计算量
    calculating_params_flops(model, universal_config.input_size_h, Mylogger_instance)

    print('Step6: Prepareing Criterion, Optimizer, Scheduler and Amp')
    # 以 超参数-criterion 配置 损失函数
    criterion = universal_config.criterion
    # 以 universal_config 为 对应模型 配置 优化器
    optimizer = get_optimizer(universal_config, model)
    # 以 universal_config 为 优化器 配置 学习策略
    scheduler = get_scheduler(universal_config, optimizer)

    # 打印 日志分割信息
    Mylogger_instance.logger.info('#----------Amp Info----------#')
    # 若 automatic_mixed_precision启用 ，定义 启用混合精度训练日志信息
    if universal_config.automatic_mixed_precision:
        log_info = '\tEnabled Automatic Mixed Precision'
    # 若 automatic_mixed_precision启用 ，定义 停用混合精度训练日志信息
    else:
        log_info = '\tDisabled Automatic Mixed Precision'
    # 打印 日志信息
    Mylogger_instance.logger.info(log_info)
    print(log_info)
    # 创建 混合精度反向传播梯度放大器，由automatic_mixed_precision决定是否启用
    grad_scaler = torch.cuda.amp.GradScaler(enabled=universal_config.automatic_mixed_precision)

    print('Step7: Set Pretrain Model')
    Mylogger_instance.logger.info('#----------Set Pretrain Model Info----------#')
    # 定义 开始轮次
    start_epoch = 1
    # 定义 训练验证阶段总耗时
    total_time = 0
    # 定义 总迭代步数step
    step = 0
    # 定义 最小损失值min_loss
    min_loss = 9999
    # 定义 最大dice值
    max_dice = 0
    # 定义 最大mIOU值
    max_mIOU = 0
    # 定义 达成最小loss值的训练轮次数
    min_loss_epoch = 0
    # 定义 达成最大dice值的训练轮次数
    max_dice_epoch = 0
    # 定义 达成最大dice值且最小loss偏差时的 训练轮次数
    max_score_epoch = 0
    # 定义 达成最大dice值且最小loss偏差时的 loss值
    max_score_loss = 9999
    # 定义 达成最大dice值且最小loss偏差时的 偏差值
    max_score_bias = 9999

    # 占位符
    max_dice_mIOU = 0
    max_dice_loss = 9999
    max_score_dice = 0
    max_score_mIOU = 0

    # 若 预训练模型路径 不为空
    if universal_config.pretrain_model_path is not None:
        # 装载预训练模型
        pretrain_model = torch.load(universal_config.pretrain_model_path,
                                    map_location=torch.device(universal_config.device))
        # 加载 预训练模型的 预训练模型训练验证阶段总耗时pretrain_model_total_time
        pretrain_model_total_time = pretrain_model['current_total_time']
        # 加载 预训练模型的 当前轮次
        current_epoch = pretrain_model['stepped_size']
        # 计算 对应的开始轮次
        start_epoch += current_epoch
        # 加载 预训练模型的 最小损失值min_loss；达成最小loss值的训练轮次数；loss值
        min_loss, min_loss_epoch, loss = pretrain_model['min_loss'], pretrain_model['min_loss_epoch'], pretrain_model[
            'current_loss']
        # 加载 预训练模型的 最大dice值；最大mIOU值；达成最大dice值的训练轮次数；
        max_dice, max_mIOU, max_dice_epoch = pretrain_model['max_dice'], pretrain_model['max_mIOU'], pretrain_model[
            'max_dice_epoch']
        # 加载 预训练模型的 最大dice值且最小loss偏差时的训练轮次数；达成最大dice值且最小loss偏差时的 loss值；达成最大dice值且最小loss偏差时的 偏差值
        max_score_epoch, max_score_loss, max_score_bias = pretrain_model['max_score_epoch'], pretrain_model[
            'max_score_loss'], pretrain_model['max_score_bias']
        # 加载 预训练模型的 参数
        model.load_state_dict(pretrain_model['model_state_dict'], strict=False)
        # 加载 预训练模型的 优化器参数
        optimizer.load_state_dict(pretrain_model['optimizer_state_dict'])
        # 加载 预训练模型的 学习策略参数
        scheduler.load_state_dict(pretrain_model['scheduler_state_dict'])

        # 日志文件与控制台打印相关信息
        Mylogger_instance.log_and_print_custom_info(
            f'\tPretrain model loaded from {universal_config.pretrain_model_path}.\n'
            f'\tPretrain model had used time: {pretrain_model_total_time / 60:.2f} minutes\n'
            f'\t\tmin loss epoch:{min_loss_epoch}, min loss:{min_loss:.4f}\n'
            f'\t\tmax dice epoch:{max_dice_epoch}, max dice:{max_dice:.4f}, max mIOU:{max_mIOU:.4f}\n'
            f'\t\tmax score epoch:{max_score_epoch}, max score loss:{max_score_loss:.4f}, max score bias: {max_score_bias:.4f}\n')
    # 若 预训练模型路径 为空
    else:
        # 设置 预训练模型训练验证阶段总耗时pretrain_model_total_time 为空
        pretrain_model_total_time = None
        # 打印 日志信息
        Mylogger_instance.log_and_print_custom_info('No pretrain model loaded', indent=True)

    print('Step8: Training')
    Mylogger_instance.logger.info('#----------Training and Validating Info----------#')
    refresh_ALASR = False
    torch.cuda.reset_peak_memory_stats(universal_config.device)
    # 以 总训练轮次长度 为 迭代,以start_epoch为起始点
    for epoch in range(start_epoch, universal_config.total_epochs + 1):
        # 记录 开始时间
        start_time = time.time()

        torch.cuda.empty_cache()
        # 执行一次 训练过程，返回 总迭代步数step
        step = train_one_epoch(train_dataloader,
                               model,
                               criterion,
                               optimizer,
                               scheduler,
                               grad_scaler,
                               epoch,
                               step,
                               Mylogger_instance,
                               universal_config)

        # 执行一次 验证过程，返回 损失值loss, 平均交并比mIOU, dice值
        loss, mIOU, dice = evaluate_one_epoch(val_dataloader,
                                              model,
                                              criterion,
                                              scheduler,
                                              epoch,
                                              Mylogger_instance,
                                              universal_config,
                                              universal_config.output_directory,
                                              validation=True
                                              )

        # 若 验证过程的loss 小于 总训练轮次中的最小loss值
        if loss < min_loss:
            refresh_ALASR = True
            # 记录 最小loss值
            min_loss = loss
            # 记录 达成最小loss值的训练轮次数
            min_loss_epoch = epoch
            # 更新 达成最大dice值且最小loss偏差时的 偏差值
            max_score_bias = max_score_loss - min_loss
            # 保存 达成最小loss值的权重文件
            save_model(total_time, epoch, min_loss, max_mIOU, max_dice, min_loss_epoch, max_dice_epoch, max_score_epoch,
                       max_score_loss, max_score_bias, loss, model, optimizer, scheduler, 'min_loss.pth',
                       universal_config)
            # 若 当前epoch / 总训练轮次长度 的比值 大于等于 验证结果信息的占总训练轮次的比值
            if (epoch / universal_config.total_epochs) >= universal_config.result_interval:
                # 打印 日志信息
                Mylogger_instance.log_and_print_custom_info(
                    f'min_loss.pth Updated: epoch:{min_loss_epoch}, loss:{min_loss:.4f}', indent=True)

        # 若 mIOU与dice 不为空
        if (mIOU is not None) and (dice is not None):
            if dice > max_dice:
                refresh_ALASR = True
                # 记录 最大dice值
                max_dice = dice
                # 记录 达成最大dice值的 训练轮次数
                max_dice_epoch = epoch
                # 记录 达成最大dice值的 loss值
                max_dice_loss = loss
                # 记录 达成最大dice值的 mIOU值
                max_dice_mIOU = mIOU
                # 保存 达成最大dice值的权重文件
                save_model(total_time, epoch, min_loss, max_mIOU, max_dice, min_loss_epoch, max_dice_epoch,
                           max_score_epoch, max_score_loss, max_score_bias, loss, model, optimizer, scheduler,
                           'max_dice.pth', universal_config)
                # 若 当前epoch / 总训练轮次长度 的比值 大于等于 验证结果信息的占总训练轮次的比值
                if (epoch / universal_config.total_epochs) >= universal_config.result_interval:
                    # 打印 日志信息
                    Mylogger_instance.log_and_print_custom_info(
                        f'max_dice.pth Updated: epoch:{max_dice_epoch}, dice:{max_dice:.4f}, mIOU:{max_dice_mIOU:.4f}, loss:{max_dice_loss:.4f}', indent=True)

                # 计算 当前的loss偏差
                loss_bias = loss - min_loss
                # 若loss偏差达到最小
                if loss_bias < max_score_bias and loss_bias != 0:
                    # 记录 达成最大dice值且最小loss偏差时的 loss值
                    max_score_loss = loss
                    # 记录 达成最大dice值且最小loss偏差时的 dice值
                    max_score_dice = dice
                    # 记录 达成最大dice值且最小loss偏差时的 偏差值
                    max_score_bias = loss_bias
                    # 记录 达成最大dice值且最小loss偏差时的 mIOU值
                    max_score_mIOU = mIOU
                    # 记录 达成最大dice值且最小loss偏差时的 训练轮次数
                    max_score_epoch = epoch
                    # 保存 达成最大dice值且最小loss偏差时的 权重文件
                    save_model(total_time, epoch, min_loss, max_mIOU, max_dice, min_loss_epoch, max_dice_epoch,
                               max_score_epoch, max_score_loss, max_score_bias, loss, model, optimizer, scheduler,
                               'max_score.pth', universal_config)

                    # 若 当前epoch / 总训练轮次长度 的比值 大于等于 验证结果信息的占总训练轮次的比值
                    if (epoch / universal_config.total_epochs) >= universal_config.result_interval:
                        # 打印 日志信息
                        Mylogger_instance.log_and_print_custom_info(
                            f'max_score.pth Updated: epoch:{max_score_epoch}, dice:{max_score_dice:.4f}, mIOU:{max_score_mIOU:.4f}, loss:{max_score_loss:.4f}, bias: {max_score_bias:.4f}', indent=True)

        if universal_config.scheduler == 'AdaptiveLinearAnnealingSoftRestarts':
            scheduler.step(loss, refresh_ALASR)
            refresh_ALASR = False

        # 保存 本次训练轮次的权重文件
        save_model(total_time, epoch, min_loss, max_mIOU, max_dice, min_loss_epoch, max_dice_epoch, max_score_epoch,
                   max_score_loss, max_score_bias, loss, model, optimizer, scheduler, 'latest.pth', universal_config)
        # torch.save(model.state_dict(), str(universal_config.best_models_directory / f'latest.pth'))

        # 计算 距离早停点轮次
        early_stopping_remaining_epoch = max(min_loss_epoch, max_dice_epoch,
                                             max_score_epoch) + universal_config.early_stopping_patience - epoch

        # 若 距离早停点轮次为0
        if early_stopping_remaining_epoch <= 0:
            # 进行早停
            Mylogger_instance.log_and_print_custom_info(f'Early Stopped! Stop at epoch{epoch}.', indent=True)
            break

        # 记录 结束时间
        end_time = time.time()
        # 计算 本次训练轮次的耗时
        epoch_time = end_time - start_time
        # 计算 从运行开始的总耗时
        total_time += epoch_time
        # 计算 剩余训练轮次
        remaining_epochs = universal_config.total_epochs - epoch
        # 若 预训练模型训练验证阶段总耗时 不为空，即有预训练模型加载
        if pretrain_model_total_time is not None:
            # 估计 本次训练验证阶段的总剩余耗时
            total_estimated_remaining_time = (total_time / (epoch - start_epoch + 1)) * remaining_epochs
        # 若 预训练模型训练验证阶段总耗时 为空，即没有预训练模型
        else:
            # 估计 训练验证阶段的总剩余耗时
            total_estimated_remaining_time = (total_time / epoch) * remaining_epochs
        # 若 预训练模型训练验证阶段总耗时 不为空，即有预训练模型加载
        if pretrain_model_total_time is not None:
            # 估计 本次训练验证阶段距离早停点剩余时间
            early_stop_estimated_remaining_time = (total_time / (
                    epoch - start_epoch + 1)) * early_stopping_remaining_epoch
        # 若 预训练模型训练验证阶段总耗时 为空，即没有预训练模型
        else:
            # 估计 距离早停点剩余时间
            early_stop_estimated_remaining_time = (total_time / epoch) * early_stopping_remaining_epoch

        # 若达到 打印剩余时间的轮次间隔
        if epoch % universal_config.estimate_interval == 0:
            # 打印 距离早停点轮次
            Mylogger_instance.log_and_print_custom_info(
                f'Before early Stop point remain {early_stopping_remaining_epoch} epoch(s)', indent=True)

            if early_stopping_remaining_epoch <= 70:
                Mylogger_instance.log_and_print_custom_info(
                    f'Supplementary Information: Max dice epoch: {max_dice_epoch}, dice:{max_dice:.4f}', indent=True)
            # 若 预训练模型训练验证阶段总耗时 不为空，即有预训练模型加载
            if pretrain_model_total_time is not None:
                # 打印 日志信息
                print(f'\tPretrain model used time: {pretrain_model_total_time / 60:.2f}minutes. '
                      f'Current training used time: {total_time / 60:.2f} minutes. Estimated '
                      f'total remaining time: {total_estimated_remaining_time / 60:.2f} minutes. Estimated remaining '
                      f'time at early stop point: {early_stop_estimated_remaining_time / 60:.2f} minutes.')
            else:
                print(
                    f'\tCurrent training used time: {total_time / 60:.2f} minutes. Estimated total remaining time: {total_estimated_remaining_time / 60:.2f} minutes. Estimated remaining time at early stop point: {early_stop_estimated_remaining_time / 60:.2f} minutes.')

    # 训练结束，记录显存使用峰值
    max_memory_allocated = torch.cuda.max_memory_allocated(universal_config.device) / 1024 ** 2  # 以MB为单位

    # 训练结束，打印 日志信息
    Mylogger_instance.log_and_print_custom_info(f'\nTraining Ends: \n' \
                                                f'\t\t Maximum CUDA memory usage:{max_memory_allocated:.2f} MB \n'
                                                f'\t\t min_loss.pth info : epoch:{min_loss_epoch}, loss:{min_loss:.4f}\n' \
                                                f'\t\t max_dice.pth info : epoch:{max_dice_epoch}, dice:{max_dice:.4f}, mIOU:{max_dice_mIOU:.4f}, loss:{max_dice_loss:.4f}\n' \
                                                f'\t\t max_score.pth info: epoch:{max_score_epoch}, dice:{max_score_dice:.4f}, mIOU:{max_score_mIOU:.4f}, loss:{max_score_loss:.4f}, bias: {max_score_bias:.4f}\n')

    print('Step9: Testing')
    Mylogger_instance.logger.info('#----------Testing Info----------#')
    if max_score_epoch == 0:
        test_model_list = ['min_loss.pth', 'max_dice.pth']
    else:
        test_model_list = ['min_loss.pth', 'max_dice.pth', 'max_score.pth']
    # 以 三个主要输出模型 为迭代
    for best_model_file_name in test_model_list:
        # 在最佳模型路径中 寻找 本次迭代对应模型文件
        best_model_file = universal_config.best_models_directory.glob(best_model_file_name)
        # 取 查找器的第一个文件
        best_model_file = next(best_model_file, None)
        # 加载模型
        best_model = torch.load(best_model_file, map_location=torch.device(universal_config.device), weights_only=False)
        model.load_state_dict(best_model['model_state_dict'], strict=False)
        # 构建 输出图片目录
        output_directory = universal_config.output_directory / best_model_file_name[:-4]
        # 若 输出图片目录 不存在，创建之
        if not output_directory.exists():
            output_directory.mkdir(parents=True, exist_ok=True)

        # 进行验证
        loss = evaluate_one_epoch(test_dataloader,
                                  model,
                                  criterion,
                                  scheduler,
                                  0,
                                  Mylogger_instance,
                                  universal_config,
                                  output_directory,
                                  validation=False,
                                  test_data_name=best_model_file_name[:-4]
                                  )
        # 若 有一致的模型，则提前终止
        if best_model_file_name == 'min_loss.pth' and min_loss_epoch == max_dice_epoch == max_score_epoch:
            break
        if best_model_file_name == 'max_dice.pth' and max_dice_epoch == max_score_epoch:
            break


def save_model(total_time, epoch, min_loss, max_mIOU, max_dice, min_loss_epoch, max_dice_epoch, max_score_epoch,
               max_score_loss, max_score_bias, loss, model, optimizer, scheduler, model_name, universal_config):
    torch.save(
        {
            'current_total_time': total_time,
            'stepped_size': epoch,
            'min_loss': min_loss,
            'max_mIOU': max_mIOU,
            'max_dice': max_dice,
            'min_loss_epoch': min_loss_epoch,
            'max_dice_epoch': max_dice_epoch,
            'max_score_epoch': max_score_epoch,
            'max_score_loss': max_score_loss,
            'max_score_bias': max_score_bias,
            'current_loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, str(universal_config.best_models_directory / model_name))


def main(execute_model_index, execute_pretrain_model_index, execute_dataset_index, criterion=None, optimizer=None,
         scheduler=None, num_workers=None):
    universal_config = UniversalConfig(execute_model_index, execute_pretrain_model_index, execute_dataset_index,
                                       criterion, optimizer, scheduler, num_workers)
    # ↓—————————————————————————————————————————————————————↓
    if universal_config.execute_model == 'CVM_UNet':
        model_config = CVM_UNetConfig(universal_config)
    elif universal_config.execute_model == 'SwinUMamba':
        model_config = SwinUMambaConfig(universal_config)

    elif universal_config.execute_model == 'CVM_UNet_FFC':
        model_config = CVM_UNetConfig(universal_config)
    elif universal_config.execute_model == 'CVM_UNet_CFF':
        model_config = CVM_UNetConfig(universal_config)
    elif universal_config.execute_model == 'CVM_UNet_Parallel':
        model_config = CVM_UNetConfig(universal_config)
    elif universal_config.execute_model == 'CVM_UNet_Independent':
        model_config = CVM_UNetConfig(universal_config)
    elif universal_config.execute_model == 'CVM_UNet_MLP':
        model_config = CVM_UNetConfig(universal_config)
    elif universal_config.execute_model == 'CVM_UNet_SE':
        model_config = CVM_UNetConfig(universal_config)
    elif universal_config.execute_model == 'CVM_UNet_FCA':
        model_config = CVM_UNetConfig(universal_config)
    # ↑—————————————————————————————————————————————————————↑
    execute(universal_config, model_config)


if __name__ == '__main__':
    universal_config = UniversalConfig(execute_model_index=8, execute_pretrain_model_index=0, execute_dataset_index=1,
                                       criterion='BceDiceLoss', optimizer=None, scheduler=None, num_workers=None)
    # ↓—————————————————————————————————————————————————————↓
    if universal_config.execute_model == 'CVM_UNet':
        model_config = CVM_UNetConfig(universal_config)
    elif universal_config.execute_model == 'SwinUMamba':
        model_config = SwinUMambaConfig(universal_config)

    elif universal_config.execute_model == 'CVM_UNet_FFC':
        model_config = CVM_UNetConfig(universal_config)
    elif universal_config.execute_model == 'CVM_UNet_CFF':
        model_config = CVM_UNetConfig(universal_config)
    elif universal_config.execute_model == 'CVM_UNet_Parallel':
        model_config = CVM_UNetConfig(universal_config)
    elif universal_config.execute_model == 'CVM_UNet_Independent':
        model_config = CVM_UNetConfig(universal_config)
    elif universal_config.execute_model == 'CVM_UNet_MLP':
        model_config = CVM_UNetConfig(universal_config)
    elif universal_config.execute_model == 'CVM_UNet_SE':
        model_config = CVM_UNetConfig(universal_config)
    elif universal_config.execute_model == 'CVM_UNet_FCA':
        model_config = CVM_UNetConfig(universal_config)
    # ↑—————————————————————————————————————————————————————↑
    execute(universal_config, model_config)
