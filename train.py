#coding:utf-8
import os
import torch
from torch import optim
from model import Net
from loss import criterion
from tools import Progbar
import config as cfg
from data_loader import get_train_dataloader, get_val_dataloader
import torch.nn as nn
import numpy as np

def ajust_learning_tri(optimizer, clr_iterations, step_size, base_lr=1e-6, max_lr=1e-3):
    cycle = np.floor(1 + clr_iterations / (2 * step_size))
    x = np.abs(clr_iterations / step_size - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) / (2 ** (cycle - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

lr_list  =[]
def train(model, optimizer, epochs, trainloader, valloader):
    model.train()
    for epoch_index in range(epochs):
        pbar = Progbar(target=len(trainloader))
        index_train = epoch_index * len(trainloader)
        running_loss = 0.0
        for batch_index, data in enumerate(trainloader):
            batch_index_ = batch_index
            batch_index_ += index_train
            lr = ajust_learning_tri(optimizer, batch_index_, step_size=len(trainloader)*2)
            # 输入数据
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            cost = criterion(outputs, labels)
            # 梯度清零
            optimizer.zero_grad()
            cost.backward()
            # 更新参数
            optimizer.step()
            running_loss += cost.item()
            pbar.update(batch_index + 1, values=[('loss', running_loss / (batch_index + 1)), ('epoch:', epoch_index)])

            lr_list.append(lr)
        val(model, valloader)
def val(model, valloader):
    correct = 0  # 预测正确的图片数
    total = 0  # 总共的图片数
    # 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        pbar = Progbar(target=len(valloader))
        for i, data in enumerate(valloader):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            cost = criterion(outputs, labels)
            running_loss += cost.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.cpu().numpy().shape[0]
            correct += (predicted.cpu().numpy() == labels.cpu().numpy()).sum()
            acc = correct / total
            pbar.update(i + 1, values=[('loss', running_loss / (i + 1)), ('acc:', acc)])
        save_model(model, acc, distributed=False)
    # print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))
def save_model(model, name_suffix, distributed=False):
    state_dict = model.module.state_dict() if distributed else model.state_dict()
    # depoly_state = {
    #     'state_dict': state_dict,
    #     'config': self.config
    # }
    # 生成后面要继续训练的模型
    filename = './model_{}.pth'.format(str(round(name_suffix,2)))
    torch.save(state_dict, filename)

def weights_add_module(weights):
    from collections import OrderedDict
    modelWeights = OrderedDict()
    for k, v in weights.items():
        name = 'module.' + k  # add `module.`
        modelWeights[name] = v
    return modelWeights

def weights_rm_module(weights):
    from collections import OrderedDict
    modelWeights = OrderedDict()
    for k, v in weights.items():
        name = k.replace('module.', '')  # remove `module.`
        modelWeights[name] = v
    return modelWeights
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    distributed = False
    if torch.cuda.device_count() > 1:
        distributed = True
        torch.cuda.set_device(0)
        # torch.distributed.init_process_group(backend="nccl", init_method='tcp://localhost:23456',
        #                                      world_size=torch.cuda.device_count(), rank=0)
        torch.distributed.init_process_group(backend="nccl")
    trainloader = get_train_dataloader(distributed)
    valloader = get_val_dataloader(distributed)
    epochs = cfg.epochs
    model = Net().cuda()
    if distributed:
        model = nn.parallel.DistributedDataParallel(model)
    # weights = weights_add_module(torch.load('./model_0.97.pth'))
    # weights = weights_rm_module(torch.load('./model_0.97.pth'))
    # model.load_state_dict(weights)#载入预训连模型
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00005)
    train(model, optimizer, epochs, trainloader, valloader)

    from matplotlib import pyplot as plt

    plt.plot(lr_list)
    plt.savefig('./show_lr.png')
if __name__ == '__main__':
    main()

