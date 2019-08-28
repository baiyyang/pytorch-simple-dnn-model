# !/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/27 11:35 PM
# @Author  : baienyang
# @Email   : baienyang@baidu.com
# @File    : leNet.py
# @Software: PyCharm
"""
Copyright 2019 Baidu, Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 超参数
EPOCH = 10
NUM_CLASSES = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据集
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                           transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root="./data", train=False,
                                          transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                                          shuffle=False)


class Net(nn.Module):
    """
    定义LeNet网络结构
    """

    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 3, 1, 2), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120), nn.BatchNorm1d(120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84), nn.BatchNorm1d(84), nn.ReLU())
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """
        前向传播
        :param x: 输入的图片矩阵
        :return: 图片的类别
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


leNet = Net(NUM_CLASSES).to(device)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(leNet.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# 训练
total_step = len(train_loader)
for epoch in range(EPOCH):
    print("epoch: {}".format(epoch))
    train_loss = 0.0
    train_acc = 0.0
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # forward pass
        outputs = leNet(batch_x)
        loss = criterion(outputs, batch_y)

        # backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch + 1, EPOCH, batch_idx + 1, total_step,
                                                                     loss.item()))

# test model
leNet.eval()
total_test = 0
with torch.no_grad():
    correct = 0
    total = 0
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        outputs = leNet(batch_x)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        total_test += len(batch_x)
    print("Test Accuracy of the model on the {} test images: {}%"
          .format(total_test, 100 * correct / total))

# save model
torch.save(leNet.state_dict(), "./model/leNet.ckpt")
