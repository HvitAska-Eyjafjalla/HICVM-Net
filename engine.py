import numpy as np
import tqdm
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


def train_one_epoch(train_dataloader, model, criterion, optimizer, scheduler, grad_scaler, epoch, step, Mylogger, config):
    # 更改为训练模式
    model.train()
    # 定义 损失值列表loss_list
    loss_list = []

    # 创建 tqdm进度条
    with tqdm.tqdm(total=len(train_dataloader), desc=f'Training[{epoch}/{config.total_epochs}]',
                   unit='Batch') as pbar:
        # 以 train_loader 迭代；iter记录 一次训练过程的迭代步数。
        for iter, data in enumerate(train_dataloader):
            # 总迭代步数step 更新
            step += iter
            # 从 data 中取出images, targets
            images, targets = data

            # 若 cuda可用 与 train_dataloader的pin_memory生效
            if config.device == 'cuda' and train_dataloader.pin_memory == 'True':
                # images, targets异步传输到cuda设备上，提升效率
                images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
            # 否则 进行普通的数据转移
            else:
                images, targets = images.to(config.device).float(), targets.to(config.device).float()

            # 前向传播
            # 若启用NVIDIA自动混合精度训练功能
            if config.automatic_mixed_precision:
                # 启用混合精度前向传播上下文管理器进行前向传播
                with autocast():
                    # 将images输入到模型中
                    out = model(images)
                    # 计算 loss值
                    loss = criterion(out, targets)
            # 若停用NVIDIA自动混合精度训练功能
            else:
                # 将images输入到模型中
                out = model(images)
                # 计算 loss值
                loss = criterion(out, targets)

            # 反向传播
            # 清除优化器中所有参数的梯度(不是置零)，以准备进行新一轮的反向传播
            optimizer.zero_grad(set_to_none=True)
            # 进行损失缩放，然后进行反向传播计算梯度
            loss.backward()
            # grad_scaler.scale(loss).backward()
            # 去除损失缩放
            # grad_scaler.unscale_(optimizer)
            # 根据超参gradient_clipping，进行梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
            # 在梯度缩放模式下，使用优化器更新模型参数
            optimizer.step()
            # grad_scaler.step()
            # 更新梯度缩放器
            # grad_scaler.update()
            # 更新进度条，更新量为一个batch_size
            pbar.update(1)

            # 向 损失值列表loss_list 添加新元素
            loss_list.append(loss.item())
            # 获取 当前优化器optimizer的学习率大小
            now_lr = optimizer.state_dict()['param_groups'][0]['lr']

            # 更新进度条后缀，显示该batch下的loss值
            pbar.set_postfix(**{'Loss (batch)': loss.item(), 'LR': now_lr})
            # 达到 于日志文件中 一次训练过程中打印一次训练信息 的轮次间隔
            if iter % config.print_interval == 0:
                log_info = f'Training[{epoch}/{config.total_epochs}]: Iter:{iter}; Loss: {loss.item():.4f}; LR: {now_lr}'
                Mylogger.logger.info(log_info)
    if config.scheduler != 'AdaptiveLinearAnnealingSoftRestarts':
        scheduler.step()
    return step


def evaluate_one_epoch(evaluate_dataloader, model, criterion, scheduler, epoch, Mylogger, config,
                       output_directory, validation, test_data_name=None):
    # 更改为验证模式
    model.eval()
    # 定义 模型的预测结果列表predictions
    predictions = []
    # 定义 真实标签值列表targets
    targets = []
    # 定义 损失值列表loss_list
    loss_list = []

    # 代码块中 不计算梯度
    with torch.no_grad():
        if validation:
            tqdm_description = f'Validating[{epoch}/{config.total_epochs}]'
        else:
            tqdm_description = f'Testing {test_data_name}'
        # 创建 tqdm进度条
        with tqdm.tqdm(total=len(evaluate_dataloader), desc=tqdm_description,
                       unit='Batch') as pbar:
            # 以 val_dataloader 迭代
            for i, data in enumerate(evaluate_dataloader):
                # 从 data 中取出images, targets
                img, msk = data

                # 若 cuda可用 与 train_dataloader的pin_memory生效
                if config.device == 'cuda' and evaluate_dataloader.pin_memory == 'True':
                    # images, targets异步传输到cuda设备上，提升效率
                    img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
                # 否则 进行普通的数据转移
                else:
                    img, msk = img.to(config.device).float(), msk.to(config.device).float()

                # 若启用NVIDIA自动混合精度训练功能
                if config.automatic_mixed_precision:
                    # 启用混合精度前向传播上下文管理器进行前向传播
                    with autocast():
                        # 将images输入到模型中
                        out = model(img)
                        # 计算 loss值
                        loss = criterion(out, msk)
                # 若停用NVIDIA自动混合精度训练功能
                else:
                    # 将images输入到模型中
                    out = model(img)
                    # 计算 loss值
                    loss = criterion(out, msk)

                # 向 损失值列表loss_list 添加新元素
                loss_list.append(loss.item())
                # 将 msk图像 添加到 真实标签值列表targets 中
                # msk图像移除 张量的第[1]个维度,并移动到CPU中，从计算图中分离张量，并转换为 NumPy 数组
                targets.append(msk.squeeze(1).cpu().detach().numpy())

                # 更新进度条，更新量为一个batch_size
                pbar.update(1)
                # 更新进度条后缀，显示该batch下的loss值
                pbar.set_postfix(**{'Loss (batch)': loss.item()})

                # 若 模型 的输出是元组
                if type(out) is tuple:
                    # 取 元组 的第一个元素作为 模型的输出
                    out = out[0]

                # out图像移除 张量的第[1]个维度,并移动到CPU中，从计算图中分离张量，并转换为 NumPy 数组
                out = out.squeeze(1).cpu().detach().numpy()
                predictions.append(out)

                # 达到 测试阶段的 保存输出图的间隔
                if i % config.save_interval == 0 and (not validation):
                    test_accuracy, test_sensitivity, test_specificity, test_Dice, test_miou, test_confusion = calculate_metrics(out.reshape(-1), msk.squeeze(1).cpu().detach().numpy().reshape(-1), config)
                    save_imgs(img, msk, out, i, test_Dice, output_directory, config.execute_dataset, config.evaluate_threshold,
                              test_data_name=test_data_name)

    # （validation）达到 全轮次中 执行并打印一次完整验证信息 的轮次间隔 或者 达到最后一次轮次
    # 或者 是测试阶段
    if (validation and (epoch % config.val_interval == 0 or epoch == config.total_epochs)) or (not validation):
        # 将 模型的预测结果列表predictions 转为 numpy数组，并展平为一维数组
        predictions = np.array(predictions).reshape(-1)
        # 将 真实标签值列表targets 转为 numpy数组，并展平为一维数组
        targets = np.array(targets).reshape(-1)
        # 计算指标
        accuracy, sensitivity, specificity, Dice, miou, confusion = calculate_metrics(predictions, targets, config)

        if validation:
            # 定义 完整验证日志信息
            log_info = f'Validation[{epoch}/{config.total_epochs}]: Loss: {np.mean(loss_list):.4f}, mIOU: {miou:.4f}, Dice: {Dice:.4f}, Accuracy: {accuracy:.4f}, Specificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}, Confusion matrix: {confusion}'
        else:
            # 定义 测试日志信息
            log_info = f'Testing {test_data_name}: loss: {np.mean(loss_list):.4f}, mIOU: {miou:.4f}, Dice: {Dice:.4f}, Accuracy: {accuracy:.4f}, Specificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}, Confusion matrix: {confusion}'
        # 打印 日志信息
        print(log_info)
        if Mylogger is not None:
            Mylogger.logger.info(log_info)
    # 若是 普通验证
    else:
        # 定义 空的miou, Dice
        miou, Dice = None, None
        # 定义 普通验证日志信息
        log_info = f'Validation[{epoch}/{config.total_epochs}]: loss: {np.mean(loss_list):.4f}'
        print(log_info)
        if Mylogger is not None:
            Mylogger.logger.info(log_info)
    return np.mean(loss_list), miou, Dice


def save_imgs(img, msk, msk_pred, i, dice, save_directory, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    msk = msk.cpu()
    img = img / 255. if img.max() > 1.1 else img
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    # 去掉第一个维度 (1, height, width) -> (height, width)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    if msk.ndim == 3 and msk.shape[0] == 1:
        msk = msk[0]
    if msk_pred.ndim == 3 and msk_pred.shape[0] == 1:
        msk_pred = msk_pred[0]

    plt.figure(figsize=(7, 15))

    plt.suptitle(f'Dice: {dice:.4f}', fontsize=16)

    plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3, 1, 2)
    plt.imshow(msk, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 1, 3)
    plt.imshow(msk_pred, cmap='gray')
    plt.axis('off')

    plt.savefig(str(save_directory / str(i)))
    plt.close()


def calculate_metrics(np_prediction, np_target, config):
    # 根据 超参evaluate_threshold 将np_prediction二值化
    y_pre = np.where(np_prediction >= config.evaluate_threshold, 1, 0)
    # 将 np_target 二值化
    y_true = np.where(np_target >= 0.5, 1, 0)

    # 计算预测标签y_pre和真实标签y_true之间的混淆矩阵confusion
    confusion = confusion_matrix(y_true, y_pre)
    # 如果只有一种标签
    if confusion.shape == (1, 1):
        # 提取出TN, FP, FN, TP
        TN, FP, FN, TP = 0, 0, 0, confusion[0, 0] if y_true[0] == 1 else confusion[0, 0]
    else:
        # 提取出TN, FP, FN, TP
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
    '''
    # 提取出TN, FP, FN, TP
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
    '''
    # 计算 准确率
    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    # 计算 灵敏度（召回率）
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    # 计算 特异性
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    # 计算 Dice值
    Dice = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    # 计算 平均交并比
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    return accuracy, sensitivity, specificity, Dice, miou, confusion
