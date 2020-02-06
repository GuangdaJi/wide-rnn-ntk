import pandas as pd
import numpy as np


class Data():
    # 更换数据集在这里改一下名字就好了
    dataset_name = 'ECG5000'
    data_dir = './data/UCRArchive_2018/' + dataset_name

    # 记录数据信息
    data_info_dir = './data/DataSummary.csv'
    info = pd.read_csv(data_info_dir)
    index = info[(info.Name == dataset_name)].index.values[0]

    train_num = min(int(info['Train '][index]), 1000)  # 训练样本也不能过长, 否则有可能爆显存...
    test_num = min(int(info['Test '][index]), 1000)

    length = int(info['Length'][index])  # length不能过长, 否则有可能爆显存

    y_dim = int(info['Class'][index])
    x_dim = 1  # 因为这个数据集的输入都是一维的

    # 读取数据, 此时数据的xy还未分开, 等训练开始时在做此处理
    train_data = np.array(pd.read_csv(data_dir + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None))[0:train_num].astype(np.float32)
    test_data = np.array(pd.read_csv(data_dir + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None))[0:test_num].astype(np.float32)


data = Data()
