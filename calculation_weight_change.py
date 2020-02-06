import torch
# from data import data
import os
# import pandas as pd
# import numpy as np
from network_config import network_config

# 此文件用于计算参数相关量, 前提是已经用相同的初始化参数, 训练了未近似和近似后的model.

# # 载入数据
# train_y = torch.tensor(data.train_data[:, 0].astype(int)).to(network_config.device) - 1
# train_x = torch.tensor(data.train_data[:, 1:]).to(network_config.device)

# test_y = torch.tensor(data.test_data[:, 0].astype(int)).to(network_config.device) - 1
# test_x = torch.tensor(data.test_data[:, 1:]).to(network_config.device)

# 载入初始化参数
init_net = torch.load(network_config.init_dir, map_location=network_config.device)

write_result = open('./model/train_log/weight_change.csv', 'w')
write_result.write('epoch, w_original - w_0, w_linear - w_0, w_original - w_linear\n')

for epoch in range(network_config.total_epoch):
    if epoch % network_config.record_interval == 0:
        # 当前epoch的模型参数保存路径
        original_net_dir = os.path.join(network_config.original_rnn_dir, 'epoch_{:0>5d}_original.pth'.format(epoch))
        linear_nn_dir = os.path.join(network_config.linear_rnn_dir, 'epoch_{:0>5d}_linear.pth'.format(epoch))

        # 载入近似与非近似的网络的结构
        original_net = torch.load(original_net_dir, map_location=network_config.device)
        linear_net = torch.load(linear_nn_dir, map_location=network_config.device)

        # 计算非近似网络参数相较于初始化参数的变化, 此公式计算的是||\theta -\theta_0||^2_2/n
        dw = (torch.pow(torch.norm(init_net.fc_hi.weight - original_net.fc_hi.weight), 2) + torch.pow(torch.norm(init_net.fc_hi.bias - original_net.fc_hi.bias), 2) + torch.pow(torch.norm(init_net.fc_hh.weight - original_net.fc_hh.weight), 2) + torch.pow(torch.norm(init_net.h0 - original_net.h0), 2) + torch.pow(torch.norm(init_net.fc_yh.weight - original_net.fc_yh.weight), 2) + torch.pow(torch.norm(init_net.fc_yh.bias - original_net.fc_yh.bias), 2)) / network_config.parameter_num

        # 计算近似网络参数相较于初始化参数的变化, 此公式计算的是||\theta_lin -\theta_0||^2_2/n
        dw_lin = (torch.pow(torch.norm(init_net.fc_hi.weight - linear_net.fc_hi.weight), 2) + torch.pow(torch.norm(init_net.fc_hi.bias - linear_net.fc_hi.bias), 2) + torch.pow(torch.norm(init_net.fc_hh.weight - linear_net.fc_hh.weight), 2) + torch.pow(torch.norm(init_net.h0 - linear_net.h0), 2) + torch.pow(torch.norm(init_net.fc_yh.weight - linear_net.fc_yh.weight), 2) + torch.pow(torch.norm(init_net.fc_yh.bias - linear_net.fc_yh.bias), 2)) / network_config.parameter_num

        # 计算非近似网络参数相较于近似网络参数的变化
        w_approx = (torch.pow(torch.norm(linear_net.fc_hi.weight - original_net.fc_hi.weight), 2) + torch.pow(torch.norm(linear_net.fc_hi.bias - original_net.fc_hi.bias), 2) + torch.pow(torch.norm(linear_net.fc_hh.weight - original_net.fc_hh.weight), 2) + torch.pow(torch.norm(linear_net.h0 - original_net.h0), 2) + torch.pow(torch.norm(linear_net.fc_yh.weight - original_net.fc_yh.weight), 2) + torch.pow(torch.norm(linear_net.fc_yh.bias - original_net.fc_yh.bias), 2)) / network_config.parameter_num

        print('epoch:{:05d}, dw:{:.10f}, dw:{:.10f}, dw:{:.30f}\n'.format(epoch, dw.item(), dw_lin.item(), w_approx.item()))
        write_result.write('{:05d}, {:.10f}, {:.10f}, {:.30f}\n'.format(epoch, dw.item(), dw_lin.item(), w_approx.item()))

        # 删除变量, 防止下一个循环时爆内存
        del original_net, linear_net, dw, dw_lin, w_approx
        torch.cuda.empty_cache()

write_result.close()
