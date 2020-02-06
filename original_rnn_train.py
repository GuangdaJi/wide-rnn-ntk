from data import data
from network_config import network_config
from original_rnn_build import Original_Net
import torch
import torch.optim as optim
import torch.nn as nn
import os

# 构建或载入初始化网络
if os.path.exists(network_config.init_dir):
    original_net = torch.load(network_config.init_dir, map_location=network_config.device)
else:
    original_net = Original_Net().to(network_config.device)
    torch.save(original_net, network_config.init_dir)

# 载入数据
train_y = torch.tensor(data.train_data[:, 0].astype(int)).to(network_config.device) - 1
train_x = torch.tensor(data.train_data[:, 1:]).to(network_config.device)

# test_y = torch.tensor(data.test_data[:, 0].astype(int)).to(network_config.device) - 1
# test_x = torch.tensor(data.test_data[:, 1:]).to(network_config.device)

# 选择初始化方法
optimizer = optim.SGD(original_net.parameters(), lr=network_config.learning_rate)
# optimizer = optim.Adagrad(original_net.parameters(), lr=network_config.learning_rate)
# optimizer = optim.Adam(original_net.parameters(), lr=network_config.learning_rate)

# 训练开始
softplus = nn.Softplus()
cross_entropy = nn.CrossEntropyLoss()
write_result = open('./model/train_log/original_net_training_dynamics.csv', 'w')

for epoch in range(network_config.total_epoch):
    optimizer.zero_grad()
    train_outputs = original_net(train_x)
    train_acc = torch.sum(torch.eq(train_y, torch.argmax(train_outputs,1)),dtype=torch.double) / data.train_num

    loss = cross_entropy(train_outputs, train_y)

    loss.backward()
    optimizer.step()    

    # test_outputs = original_net(test_x)
    # test_acc = torch.sum(torch.eq(test_y, torch.argmax(test_outputs,1)),dtype=torch.double) / data.test_num

    write_result.write('epoch:{:05d}, loss:{:.10f}, train_acc:{:.5f}'.format(epoch, loss.item(), train_acc.item()))

    if epoch % network_config.record_interval == 0:
        print('epoch:{:d}, loss:{:.10f}, train_acc:{:.5f}'.format(epoch, loss.item(), train_acc.item()))
        torch.save(original_net, os.path.join(network_config.original_rnn_dir, 'epoch_{:0>5d}_original.pth'.format(epoch)))

write_result.close()
