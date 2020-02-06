import torch
from data import data
import os
# import pandas as pd
# import numpy as np
from network_config import network_config

# 此文件用于计算empirical_kernel, 前提是已经训练了未近似的model.
# empirical_kernel是一个data_num x data_num的矩阵, 无法显示出来, 所以存到文件里

# 载入数据
# train_y = torch.tensor(data.train_data[:, 0].astype(int)).to(network_config.device) - 1
train_x = torch.tensor(data.train_data[:, 1:]).to(network_config.device)

# test_y = torch.tensor(data.test_data[:, 0].astype(int)).to(network_config.device) - 1
test_x = torch.tensor(data.test_data[:, 1:]).to(network_config.device)

# # 载入初始化参数
# init_net = torch.load(network_config.init_dir, map_location=network_config.device)

if not os.path.exists('./model/train_log/empirical_kernel'):
    os.mkdir('./model/train_log/empirical_kernel')
# 输出的是 empirical_kernel_m - empirical_kernel_0 的范数
write_result = open('./model/train_log/output_approx.csv', 'w')
write_result.write('epoch, theta_m - theta_0\n')

# specify i, j, 因为算整个empirical kernel太耗时间了, 所以我们选择特定的两个元素计算吧. 这里只给出一个例子i, j = 0, 1
i, j = 4, 1
for epoch in range(network_config.total_epoch):
    if epoch % network_config.record_interval == 0:
        # 当前epoch的模型参数保存路径
        original_net_dir = os.path.join(network_config.original_rnn_dir, 'epoch_{:0>5d}_original.pth'.format(epoch))
        # linear_nn_dir = os.path.join(network_config.linear_rnn_dir, 'epoch_{:0>5d}_linear.pth'.format(epoch))
        # 载入近似与非近似的网络的结构
        original_net = torch.load(original_net_dir, map_location=network_config.device)
        # linear_net = torch.load(linear_nn_dir, map_location=network_config.device)

        # 计算empirical kernel
        y = original_net(train_x)  # N x C
        # 因为pytorch只能对标量求导,所以要把上面的y矢量构造成一个标量
        temp = torch.ones(data.train_num, 1, dtype=torch.float, device=network_config.device, requires_grad=True)  # N x 1
        t = torch.mm(temp, torch.ones(1, data.y_dim, dtype=torch.float, device=network_config.device, requires_grad=False))  # 1 x C

        fc_hi_w_grad = torch.autograd.grad(y, original_net.fc_hi.weight, grad_outputs=t, create_graph=True)[0]
        fc_hi_b_grad = torch.autograd.grad(y, original_net.fc_hi.bias, grad_outputs=t, create_graph=True)[0]

        fc_hh_w_grad = torch.autograd.grad(y, original_net.fc_hh.weight, grad_outputs=t, create_graph=True)[0]

        fc_yh_w_grad = torch.autograd.grad(y, original_net.fc_yh.weight, grad_outputs=t, create_graph=True)[0]
        fc_yh_b_grad = torch.autograd.grad(y, original_net.fc_yh.bias, grad_outputs=t, create_graph=True)[0]

        h0_grad = torch.autograd.grad(y, original_net.h0, grad_outputs=t, create_graph=True)[0]

        g = torch.sum(torch.mul(fc_hi_w_grad, fc_hi_w_grad)) + torch.sum(torch.mul(fc_hi_b_grad, fc_hi_b_grad)) + torch.sum(torch.mul(fc_hh_w_grad, fc_hh_w_grad)) + torch.sum(torch.mul(fc_yh_w_grad, fc_yh_w_grad)) + torch.sum(torch.mul(fc_yh_b_grad, fc_yh_b_grad)) + torch.sum(torch.mul(h0_grad, h0_grad))

        dg = torch.autograd.grad(g, temp, create_graph=True)[0][j, 0]
        # print(dg)
        theta = torch.autograd.grad(dg, temp)[0][i, 0]

        print('epoch:{:05d}, theta:{:.10f}'.format(epoch, theta.item()))
        write_result.write('{:05d}, {:.10f}'.format(epoch, theta.item()))

        del original_net, y, temp, fc_hi_w_grad, fc_hi_b_grad, fc_hh_w_grad, fc_yh_w_grad, fc_yh_b_grad, h0_grad, g, dg, theta
        torch.cuda.empty_cache()

write_result.close()
