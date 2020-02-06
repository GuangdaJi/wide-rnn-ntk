import torch
from data import data
import os
import torch.nn.functional as F
# import pandas as pd
# import numpy as np
from network_config import network_config

# 此文件用于计算近似网络和非近似网络的输出函数的差别, 前提是已经用相同的初始化参数, 训练了未近似和近似后的model.

# 载入数据
# train_y = torch.tensor(data.train_data[:, 0].astype(int)).to(network_config.device) - 1
train_x = torch.tensor(data.train_data[:, 1:]).to(network_config.device)

# test_y = torch.tensor(data.test_data[:, 0].astype(int)).to(network_config.device) - 1
test_x = torch.tensor(data.test_data[:, 1:]).to(network_config.device)

# # 载入初始化参数
# init_net = torch.load(network_config.init_dir, map_location=network_config.device)

write_result = open('./model/train_log/output_approx.csv', 'w')
write_result.write('epoch, f_original - f_linear @train, f_original - f_linear @test\n')

for epoch in range(network_config.total_epoch):
    if epoch % network_config.record_interval == 0:
        # 当前epoch的模型参数保存路径
        original_net_dir = os.path.join(network_config.original_rnn_dir, 'epoch_{:0>5d}_original.pth'.format(epoch))
        linear_nn_dir = os.path.join(network_config.linear_rnn_dir, 'epoch_{:0>5d}_linear.pth'.format(epoch))
        # 载入近似与非近似的网络的结构
        original_net = torch.load(original_net_dir, map_location=network_config.device)
        linear_net = torch.load(linear_nn_dir, map_location=network_config.device)
        # 计算输出
        original_train_outputs = F.softmax(original_net(train_x), dim=1)
        linear_train_outputs = F.softmax(linear_net(train_x), dim=1)
        original_test_outputs = F.softmax(original_net(test_x), dim=1)
        linear_test_outputs = F.softmax(linear_net(test_x), dim=1)

        # 计算输出的差别
        train_approx = torch.sum(torch.norm(original_train_outputs - linear_train_outputs, dim=1)) / data.train_num
        test_approx = torch.sum(torch.norm(original_test_outputs - linear_test_outputs, dim=1)) / data.test_num

        print('epoch:{:05d}, train_approx:{:.10f}, test_approx:{:.10f}'.format(epoch, train_approx.item(), test_approx.item()))
        write_result.write('{:05d}, {:.10f}, {:.10f}'.format(epoch, train_approx.item(), test_approx.item()))

        del original_net, linear_net, original_train_outputs, linear_train_outputs, original_test_outputs, linear_test_outputs, train_approx, test_approx
        torch.cuda.empty_cache()

write_result.close()
