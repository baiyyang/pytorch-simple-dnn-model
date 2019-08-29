# !/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/29 11:20 PM
# @Author  : baienyang
# @Email   : baienyang@baidu.com
# @File    : linear_regression.py
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
import numpy as np
import matplotlib.pyplot as plt

# 超参数
INPUT_SIZE = 1
OUTPUT_SIZE = 1
EPOCHS = 60
LEARNING_RATE = 0.001

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

linear_model = nn.Linear(INPUT_SIZE, OUTPUT_SIZE)

# 优化器和损失函数
optimizer = torch.optim.SGD(linear_model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    # forward pass
    outputs = linear_model(inputs)
    loss = criterion(labels, outputs)
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, EPOCHS, loss.item()))


# plot the model
predicted = linear_model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

# save model
torch.save(linear_model.state_dict(), "./model/linear_model.ckpt")
