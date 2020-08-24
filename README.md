# pytorch_sample_project
pytorch的简单工程实现，数据集采用的是mnist，带有单卡和分布式多卡示例.
其中train.py，里面环境变量os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'，设置为多张卡，则执行sh multi_gpu_train.sh进行训练,注意修改.sh里面的nproc_per_node为卡的数量;
如果设置为单卡，执行python train.py即可.
数据增强采用imgaug，https://imgaug.readthedocs.io/en/latest/source/overview/color.html#changecolortemperature
学习率采用周期性衰减。
