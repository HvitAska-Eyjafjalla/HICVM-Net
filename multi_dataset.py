import numpy as np
import os
from PIL import Image

import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image

import tqdm
from PIL import Image

from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2


class MultiDataset(Dataset):
    def __init__(self, universal_config, Mylogger=None, train=False, validation=False, test=False):
        super(MultiDataset, self).__init__()
        self.universal_config = universal_config
        # 若 对应数据集的子集 是 训练集
        if train:
            sub_directory_name = 'Train_Folder'
        # 若 对应数据集的子集 是 验证集
        elif validation:
            sub_directory_name = 'Val_Folder'
        # 若 对应数据集的子集 是 测试集
        elif test:
            sub_directory_name = 'Test_Folder'
        else:
            assert 'error'

        # 附加条件
        # 若 对应数据集为'Glas'数据集 ，验证集与测试集 统一采用测试集
        if self.universal_config.execute_dataset == 'Glas' and validation:
            sub_directory_name = 'Test_Folder'

        # 定义 对应数据集的子集目录sub_directory
        sub_directory = self.universal_config.dataset_directory / sub_directory_name
        # 定义 对应数据集的子集-img子目录image_directory
        image_directory = sub_directory / 'img'
        # 定义 对应数据集的子集-mask子目录mask_directory
        mask_directory = sub_directory / 'labelcol'

        # 将 image_directory目录下全部文件名 添加到 列表images_list
        images_list = sorted([f.name for f in list(image_directory.iterdir()) if
                              f.is_file() and f.suffix.lower().lstrip('.') in ['jpg', 'jpeg', 'png', 'bmp', 'tif']])
        # 将 mask_directory目录下全部文件名 添加到 列表images_list
        masks_list = sorted([f.name for f in list(mask_directory.iterdir()) if
                             f.is_file() and f.suffix.lower().lstrip('.') in ['jpg', 'jpeg', 'png', 'bmp', 'tif']])

        # 获取 image_directory全部文件 的均值与标准差
        if universal_config.execute_dataset == 'ISIC2018':
            if train:
                self.mean = np.array([180.649657415565968676673946902155876159667968750000,
                                      148.441095474935650599945802241563796997070312500000,
                                      136.695205547370107979077147319912910461425781250000])
                self.std = np.array([24.934896689917430023797351168468594551086425781250,
                                     28.744171859365177823519843514077365398406982421875,
                                     32.379201570118581798851664643734693527221679687500])
            elif validation:
                self.mean = np.array([180.971604079981801760368398390710353851318359375000,
                                      137.769935441633577966058510355651378631591796875000,
                                      126.338713838523830190752050839364528656005859375000])
                self.std = np.array([28.555279284538805484316981164738535881042480468750,
                                     33.000485090765970142001606291159987449645996093750,
                                     37.215216263719099742957041598856449127197265625000])
            elif test:
                self.mean = np.array([180.207008616429646963297273032367229461669921875000,
                                      137.505068890689727822973509319126605987548828125000,
                                      126.596198364011513604054925963282585144042968750000])
                self.std = np.array([26.351504992519981129817097098566591739654541015625,
                                     31.112553580482529724804408033378422260284423828125,
                                     35.428664979540684498715563677251338958740234375000])
        else:
            self.mean, self.std = self.get_dataset_mean_std(images_list, image_directory, sub_directory_name)

        # 定义 日志信息
        log_info = f'Created dataset from {sub_directory_name}, length:{len(images_list)}, mean:{self.mean}, std:{self.std}'
        print(log_info)
        # 向 日志对象Mylogger 添加 日志信息log_info
        if Mylogger is not None:
            Mylogger.logger.info(log_info)

        # 定义 二维列表data，第一维存储image-mask列表对，第二维存储各个列表对
        self.data = []
        # 以 images_list长度 迭代
        for i in range(len(images_list)):
            # 获取 本次迭代的 对应image的文件路径images_path
            images_path = image_directory / images_list[i]
            # 获取 本次迭代的 对应mask的文件路径mask_path
            mask_path = mask_directory / masks_list[i]
            # 二维列表data 新增元素 image-mask列表对
            self.data.append([images_path, mask_path])

        # 若 对应数据集的子集 是 训练集
        if train:
            # 定义 变换过程transformer 为 训练集变换过程
            self.transformer = transforms.Compose([
                # 图像归一化
                # myNormalize(mean=self.mean, std=self.std),
                # 50%概率进行翻转
                # myRandomFlip(p=0.5),
                # 50%的概率进行水平翻转（数据增强）
                # myRandomHorizontalFlip(p=0.5),
                # 50%的概率进行垂直翻转（数据增强）
                # myRandomVerticalFlip(p=0.5),
                # 50%的概率进行[0, 360]的旋转 概率出现全零张量 已经弃用
                # myRandomRotation(p=0.5),
                # 转换为张量格式
                # myToTensor(),
                # 重新调整图片尺寸以统一化
                # myResize(self.universal_config.input_size_h, self.universal_config.input_size_w)

                myResize(self.universal_config.input_size_h, self.universal_config.input_size_w),
                myRandomFlip(p=0.5),
                myRandomRotation(p=0.5),
                myNormalize(mean=self.mean, std=self.std),
                myToTensor()
            ])
        # 若 对应数据集的子集 是 验证集或者测试集
        else:
            # 定义 变换过程transformer 为 验证集或者测试集变换过程
            self.transformer = transforms.Compose([
                # 图像归一化
                # myNormalize(mean=self.mean, std=self.std),
                # 转换为张量格式
                # myToTensor(),
                # 重新调整图片尺寸以统一化
                # myResize(self.universal_config.input_size_h, self.universal_config.input_size_w)

                myResize(self.universal_config.input_size_h, self.universal_config.input_size_w),
                myNormalize(mean=self.mean, std=self.std),
                myToTensor()
            ])

    def __getitem__(self, index):
        # 取出 image-mask列表对
        img_path, msk_path = self.data[index]
        # 将 image 转化为 RGB格式
        img = np.array(Image.open(img_path).convert('RGB'))
        # 将 mask 转化为 灰度图格式
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2)
        if self.universal_config.execute_dataset == 'Glas':
            split_value = 1
        # elif self.universal_config.execute_dataset == 'Kvasir_SEG':
        else:
            split_value = 10
        msk[msk < split_value] = 0
        msk[msk >= split_value] = 1
        # 作 对应变换
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)

    def get_dataset_mean_std(self, images_list, image_directory, sub_directory_name):
        # 定义 image通道数
        image_channels = 3
        # 定义 全零数组
        cumulative_mean = np.zeros(image_channels)
        cumulative_std = np.zeros(image_channels)

        # 以 列表images_list 迭代，并添加tqdm进度条
        for image_name in tqdm.tqdm(images_list, total=len(images_list),
                                    desc=f'Calculating mean and std in {sub_directory_name}'):
            # 定义 本次迭代的 图片路径image_path
            image_path = image_directory / image_name
            # 打开对应 图片路径image_path 的图片
            image = np.array(Image.open(image_path))
            # 对每个维度进行统计，Image.open打开的是HWC格式，最后一维是通道数

            for d in range(3):
                cumulative_mean[d] += image[:, :, d].mean()
                cumulative_std[d] += image[:, :, d].std()

        # 对 image_directory目录下的全部文件 求总和均值与总和平均差
        mean = cumulative_mean / len(images_list)
        std = cumulative_std / len(images_list)
        return mean, std


class myToTensor:
    def __init__(self):
        pass

    def __call__(self, data):
        # image, mask = data
        # 将img与mask图像（C,H,W格式）转换为张量格式（H,W,C），以便于后续的图像预处理
        # return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask).permute(2, 0, 1)
        
        image, mask = data
        image = torch.tensor(image)
        mask = torch.tensor(mask)
        
        # 处理图像
        if image.ndimension() == 2:
            image = image.unsqueeze(0)  # 添加通道维度，变为 [1, H, W]
        elif image.ndimension() == 3:
            image = image.permute(2, 0, 1)  # 从 [H, W, C] 变为 [C, H, W]
        else:
            raise ValueError(f"Unexpected number of dimensions for image: {image.ndimension()}")

        # 处理掩码
        if mask.ndimension() == 2:
            mask = mask.unsqueeze(0)  # 添加通道维度，变为 [1, H, W]
        elif mask.ndimension() == 3:
            mask = mask.permute(2, 0, 1)  # 从 [H, W, C] 变为 [C, H, W]
        else:
            raise ValueError(f"Unexpected number of dimensions for mask: {mask.ndimension()}")
        return image, mask

'''
class myResize:
    def __init__(self, size_h=512, size_w=512):
        self.size_h = size_h
        self.size_w = size_w

    # 将 image与mask 调整为 [size_h, size_w]，默认512*512
    def __call__(self, data):
        image, mask = data
        return (TF.resize(image, [self.size_h, self.size_w]),
                TF.resize(mask, [self.size_h, self.size_w]))
'''
                
class myResize:
    def __init__(self, size_h=512, size_w=512):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask = data
        # 使用 OpenCV 进行高效的调整大小
        image_resized = cv2.resize(image, (self.size_w, self.size_h), interpolation=cv2.INTER_NEAREST)
        mask_resized = cv2.resize(mask, (self.size_w, self.size_h), interpolation=cv2.INTER_NEAREST)
        return image_resized, mask_resized


class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        # 若 随机数小于0.5 ，水平翻转image与mask
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(mask)
        else:
            return image, mask


class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        # 若 随机数小于0.5 ，垂直翻转image与mask
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(mask)
        else:
            return image, mask


class myRandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def random_rot_flip(self, image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label

    def __call__(self, data):
        image, mask = data
        # 若 随机数小于0.5 ，垂直翻转image与mask
        if random.random() < self.p:
            return self.random_rot_flip(image, mask)
        else:
            return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0, 360]):
        # self.angle = random.uniform(degree[0], degree[1])
        self.p = p

    def random_rotate(self, image, label):
        angle = np.random.randint(20, 80)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label

    def __call__(self, data):
        image, mask = data
        # 若 随机数小于0.5 ，旋转image与mask以 随机的angle角度
        if random.random() < self.p:
            # return TF.rotate(image, self.angle), TF.rotate(mask, self.angle)
            return self.random_rotate(image, mask)
        else:
            return image, mask

    # 对原始数据进行归一化处理


f'''
class myNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        # 从data取出img与mask
        img, msk = data
        # 对img进行归一化
        img_normalized = (img - self.mean) / self.std
        img_normalized = ((img_normalized - np.min(img_normalized)) / (
                np.max(img_normalized) - np.min(img_normalized)))/255
        return img_normalized, msk
'''


class myNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        # 从data取出img与mask
        img, msk = data
        # 对img进行归一化
        img_normalized = (img - np.mean(img)) / np.std(img)
        img_normalized = ((img_normalized - np.min(img_normalized)) / (np.max(img_normalized) - np.min(img_normalized)))*255
        return img_normalized, msk


