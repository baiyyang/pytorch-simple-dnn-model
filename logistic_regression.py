# !/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/29 11:46 PM
# @Author  : baienyang
# @Email   : baienyang@baidu.com
# @File    : logistic_regression.py
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

# Hyper parameter
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.01
INPUT_SIZE = 784
OUTPUT_SIZE = 10

# train and test data
train_data = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

# model
lr_model = nn.Linear(INPUT_SIZE, OUTPUT_SIZE)

# optimizer and criterion
optimizer = torch.optim.Adam(lr_model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

total_step = len(train_loader)
for epoch in range(EPOCHS):
    for index, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.reshape(-1, INPUT_SIZE)
        # forward pass
        predict = lr_model(batch_x)
        loss = criterion(predict, batch_y)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (index + 1) % 100 == 0:
            print("Epoch: [{}/{}]， Step: [{}/{}], Loss: {:.4f}"
                  .format(epoch, EPOCHS, index + 1, total_step, loss.item()))

# Test model
with torch.no_grad():
    correct = 0
    total = 0
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.reshape(-1, INPUT_SIZE)
        predict = lr_model(batch_x)
        _, predict = torch.max(predict.data, 1)
        total += batch_y.size(0)
        correct += (predict == batch_y).sum()
    print("Accuracy of the model on the {} test images: {:.4f}%".format(total, 100 * correct / total))

# Save model
torch.save(lr_model.state_dict(), "./model/lr_model.ckpt")

