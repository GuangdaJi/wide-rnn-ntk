import torch.nn as nn
import torch
import os
from network_config import network_config
from original_rnn_build import Original_Net

# 构建初始化的网络结构
if os.path.exists(network_config.init_dir):
    init_net = torch.load(network_config.init_dir, map_location=network_config.device)
else:
    init_net = Original_Net().to(network_config.device)
    torch.save(init_net, network_config.init_dir)


# 定义线性近似后的网络结构
class Linear_Net(nn.Module):
    def __init__(self):
        super(Linear_Net, self).__init__()

        # 输入x_t到隐变量h_t的全连接层
        self.fc_hi = nn.Linear(network_config.input_dim, network_config.hidden_layer_dim, bias=True)
        self.fc_hi.weight = init_net.fc_hi.weight
        self.fc_hi.bias = init_net.fc_hi.bias

        # 隐变量h_{t-1}到隐变量h_t的全连接层
        self.fc_hh = nn.Linear(network_config.hidden_layer_dim, network_config.hidden_layer_dim, bias=False)
        self.fc_hh.weight = init_net.fc_hh.weight

        # 隐变量h_T到隐输出的全连接层
        self.fc_yh = nn.Linear(network_config.hidden_layer_dim, network_config.output_dim, bias=True)
        self.fc_yh.weight = init_net.fc_yh.weight
        self.fc_yh.bias = init_net.fc_yh.bias

        # 初始隐变量h_0
        self.h0 = nn.Parameter(init_net.h0)

    def forward(self, x):
        y0 = init_net(x)  # 零阶项

        # 下面是求导过程
        temp = torch.ones(y0.size(), dtype=torch.float, device=network_config.device, requires_grad=True)

        fc_hi_w_grad = torch.autograd.grad(y0, init_net.fc_hi.weight, grad_outputs=temp, create_graph=True)[0]
        fc_hi_b_grad = torch.autograd.grad(y0, init_net.fc_hi.bias, grad_outputs=temp, create_graph=True)[0]

        fc_hh_w_grad = torch.autograd.grad(y0, init_net.fc_hh.weight, grad_outputs=temp, create_graph=True)[0]

        fc_yh_w_grad = torch.autograd.grad(y0, init_net.fc_yh.weight, grad_outputs=temp, create_graph=True)[0]
        fc_yh_b_grad = torch.autograd.grad(y0, init_net.fc_yh.bias, grad_outputs=temp, create_graph=True)[0]

        h0_grad = torch.autograd.grad(y0, init_net.h0, grad_outputs=temp, create_graph=True)[0]

        # 加在一起就很麻烦...
        g = torch.sum(torch.mul(fc_hi_w_grad, self.fc_hi.weight - init_net.fc_hi.weight)) + torch.sum(torch.mul(fc_hi_b_grad, self.fc_hi.bias - init_net.fc_hi.bias)) + torch.sum(torch.mul(fc_hh_w_grad, self.fc_hh.weight - init_net.fc_hh.weight)) + torch.sum(torch.mul(fc_yh_w_grad, self.fc_yh.weight - init_net.fc_yh.weight)) + torch.sum(torch.mul(fc_yh_b_grad, self.fc_yh.bias - init_net.fc_yh.bias)) + torch.sum(torch.mul(h0_grad, self.h0 - init_net.h0))

        dy = torch.autograd.grad(g, temp, create_graph=True)[0]
        y = y0 + dy
        return y


linear_net = Linear_Net()
