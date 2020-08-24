distributed = True
batch_size = 64
epochs =100
# def ajust_learning_tri(clr_iterations, step_size, base_lr=1e-6, max_lr=1e-3):
#     cycle = np.floor(1 + clr_iterations / (2 * step_size))
#     x = np.abs(clr_iterations / step_size - 2 * cycle + 1)
#     lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) / (2 ** (cycle - 1))
#     # for param_group in optimizer.param_groups:
#     #     param_group['lr'] = lr
#     return lr
#
# import numpy as np
# epochs = 10
#
# lr_list = []
# for epoch_index in range(epochs):
#     # pbar = Progbar(target=len(trainloader))
#     index_train = epoch_index * (839)
#     running_loss = 0.0
#     for batch_index in range(839):
#
#         batch_index += index_train
#         lr = ajust_learning_tri(batch_index, step_size=839)
#         lr_list.append(lr)
#
# print(lr_list)
# from matplotlib import pyplot as plt
# plt.plot(lr_list)
# # plt.savefig('./show.png')
# plt.show()
#
# import numpy as np
# from matplotlib import pyplot as plt
# lr_list = []
# step_size = 1000
# base_lr = 1e-6
# max_lr = 1e-3
#
# for clr_iterations in range(8000):
#     cycle = np.floor(1 + clr_iterations / (2 * step_size))
#     x = np.abs(clr_iterations / step_size - 2 * cycle + 1)
#     lr=base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))/(2 ** (cycle - 1))
#     lr_list.append(lr)
#
# plt.plot(lr_list,'r')
# plt.show()