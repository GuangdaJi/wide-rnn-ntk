import torch.nn as nn
import torch.nn.functional as F
import torch
from data import data
import numpy as np
from network_config import network_config


# 定义未近似的网络结构
class Original_Net(nn.Module):
    def __init__(self):
        super(Original_Net, self).__init__()

        # 输入x_t到隐变量h_t的全连接层
        self.fc_hi = nn.Linear(network_config.input_dim, network_config.hidden_layer_dim, bias=True)
        nn.init.normal_(self.fc_hi.weight.data, std=network_config.weight_std / np.sqrt(network_config.input_dim))  # 根据高斯分布初始化
        nn.init.normal_(self.fc_hi.bias.data, std=network_config.bias_std)

        # 隐变量h_{t-1}到隐变量h_t的全连接层
        self.fc_hh = nn.Linear(network_config.hidden_layer_dim, network_config.hidden_layer_dim, bias=False)
        nn.init.normal_(self.fc_hh.weight.data, std=network_config.weight_std / np.sqrt(network_config.hidden_layer_dim))

        # 隐变量h_T到隐输出的全连接层
        self.fc_yh = nn.Linear(network_config.hidden_layer_dim, network_config.output_dim, bias=True)
        nn.init.normal_(self.fc_yh.weight.data, std=network_config.weight_std / np.sqrt(network_config.hidden_layer_dim))
        nn.init.normal_(self.fc_yh.bias.data, std=network_config.bias_std)

        # 初始隐变量h_0
        self.h0 = nn.Parameter(torch.randn(1, network_config.hidden_layer_dim))

    def forward(self, x):
        h = torch.mm(torch.ones(x.size()[0], 1, device=network_config.device), self.h0)
        for t in range(data.length):
            h = F.relu(self.fc_hi(x[:, t:t + 1]) + self.fc_hh(h))  # 循环带入
        y = self.fc_yh(h)
        return y
