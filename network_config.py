import torch
from data import data
import os


class Network_Config():
    # 网络结构参数
    input_dim = data.x_dim
    output_dim = data.y_dim
    hidden_layer_dim = 1000

    # 初始化参数的标准差
    weight_std = 1.0
    bias_std = 1.0

    # 训练相关参数
    learning_rate = 0.0005
    total_epoch = 50
    record_interval = 10  # 存储模型的间隔
    parameter_num = input_dim * hidden_layer_dim + hidden_layer_dim + hidden_layer_dim * hidden_layer_dim + output_dim * hidden_layer_dim + output_dim + hidden_layer_dim

    # 训练用设备
    device = torch.device('cuda:0')  # 显卡的序号, 通过命令行下 nvidia-smi查看

    # 网络结构保存路径
    init_dir = './model/init_model.pth'  # 初始化的网络结构保存路径, 原网络和近似后网络都会用到同一个初始化参数, 所以单独保存
    original_rnn_dir = './model/original_model'  # 训练过程中的网络结构保存路径, 注意这是一个文件夹, 因为要保存不同epoch的参数
    linear_rnn_dir = './model/linear_model'  # 训练过程中的线性近似网络结构保存路径, 也是一个文件夹
    train_log_dir = './model/train_log'

    if not os.path.exists('./model'):
        os.mkdir('./model')
    if not os.path.exists(original_rnn_dir):
        os.mkdir(original_rnn_dir)
    if not os.path.exists(linear_rnn_dir):
        os.mkdir(linear_rnn_dir)
    if not os.path.exists(train_log_dir):
        os.mkdir(train_log_dir)


network_config = Network_Config()
