#coding:utf-8
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset
import imgaug as ia
from imgaug import augmenters as iaa
import config as cfg
import numpy as np
show = ToPILImage()# 可以把Tensor转成Image，方便可视化

class MNISTDataset(Dataset):
    def __init__(self, datadir=None, labelsdir=None, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.data, self.targets = torch.load(datadir)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        img = img.numpy()
        img = np.concatenate((np.expand_dims(img, axis=-1), np.expand_dims(img, axis=-1), np.expand_dims(img, axis=-1)), axis=-1)
        img = self.aug(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def aug(self, img):
        seq = iaa.Sequential([
            # iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect keypoints
            # iaa.Fliplr(0.5),
            iaa.Affine(
                rotate=(0, 10),  # 0~360随机旋转
                # scale=(0.7, 1.0),#通过增加黑边缩小图片
            ),  # rotate by exactly 0~360deg and scale to 70-100%, affects keypoints
            iaa.GaussianBlur(
                sigma=(0, 3.)
            ),
            # iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
            # iaa.WithChannels(channels=0, children=iaa.Add((50, 100))),
            # iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB"),
            iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.3, 0.9)),
            iaa.MultiplyHue(mul=(0.5, 1.5))
            # iaa.Resize(0.5, 3)
        ])
        seq_def = seq.to_deterministic()
        image_aug = seq_def.augment_image(img)
        return image_aug

path = './data'
if not os.path.exists(path):
    os.mkdir(path)
# 定义对数据的预处理
transform = transforms.Compose([
        transforms.ToTensor(),# 转为Tensor 归一化至0～1
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),# 归一化
                             ])

def get_train_dataloader(distributed):
    # 训练集
    # trainset = torchvision.datasets.MNIST(
    #                     root=path,
    #                     train=True,
    #                     download=True,
    #                     transform=transform)
    trainset = MNISTDataset(datadir='./data/MNIST/processed/training.pt', transform=transform)
    train_sampler = None
    shuffle = True
    pin_memory = False
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        # 3）使用DistributedSampler
        train_sampler = DistributedSampler(trainset)
        shuffle = False
        pin_memory = True
    trainloader = torch.utils.data.DataLoader(
                        trainset,
                        sampler=train_sampler,
                        pin_memory=pin_memory,
                        batch_size=cfg.batch_size,
                        shuffle=shuffle,
                        num_workers=0)
    return trainloader

def get_val_dataloader(distributed):
    # valset = torchvision.datasets.MNIST(
    #                     path,
    #                     train=False,
    #                     download=True,
    #                     transform=transform)
    valset = MNISTDataset(datadir='./data/MNIST/processed/test.pt',transform=transform)

    val_sampler = None
    pin_memory = False
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        # 3）使用DistributedSampler
        val_sampler = DistributedSampler(valset)
        pin_memory = True

    valloader = torch.utils.data.DataLoader(
                        valset,
                        sampler=val_sampler,
                        pin_memory=pin_memory,
                        batch_size=cfg.batch_size,
                        shuffle=False,
                        num_workers=0)
    return valloader

def vis_data_cv2():
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    (data, label) = trainset[100]
    print('==data.shape:', data.shape)
    print('=classes[label]:', classes[label])
    new_data = data.numpy()
    new_data = (new_data * 0.5 + 0.5) * 255
    print(new_data.shape)
    new_data = new_data.transpose((1, 2, 0))
    cv2.imwrite('1.jpg', new_data)
    print('==len(trainloader):', len(trainloader))

def vis_data_mutilpy():
    output_path = './查看图片'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # trainloader = get_train_dataloader(distributed=False)
    valloader = get_val_dataloader(distributed=False)
    for batch_index, data in enumerate(valloader):
        if batch_index < 1:
            inputs, labels = data
            print(inputs.shape)
            print(labels)
            for j in range(inputs.shape[0]):
                input_ = inputs[j].numpy().transpose((1, 2, 0))
                img = input_*0.5 + 0.5
                label = labels[j].numpy()
                cv2.imwrite(os.path.join(output_path, str(j) + '_' + str(label)+'.jpg'), img)
            break
if __name__ == '__main__':
    # vis_data_cv2()
    vis_data_mutilpy()