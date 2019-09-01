# !/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/1 8:19 PM
# @Author  : baienyang
# @Email   : baienyang@baidu.com
# @File    : recurrent_neural_network.py
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

# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters
SEQUENCE_LENGTH = 28
INPUT_SIZE = 28
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 10
BATCH_SIZE = 100
NUM_EPOCHS = 2
LEARNING_RATE = 0.01

# data set
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# refine RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = HIDDEN_SIZE
        self.num_layers = NUM_LAYERS
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward pass, out shape: [batch_size, seq_length, hidden_length]
        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out


rnn_model = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=LEARNING_RATE)

# Train model
total_step = len(train_loader)
for epoch in range(NUM_EPOCHS):
    for index, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.reshape(-1, SEQUENCE_LENGTH, INPUT_SIZE).to(device)
        batch_y = batch_y.to(device)

        # forward pass
        outputs = rnn_model(batch_x)
        loss = criterion(outputs, batch_y)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (index + 1) % 100 == 0:
            print("EPOCH: [{}/{}], STEP: [{}/{}], LOSS: {:.4f}".format(epoch, NUM_EPOCHS, index + 1,
                                                                       total_step, loss.item()))

# Test model
with torch.no_grad():
    correct = 0
    total = 0
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.reshape(-1, SEQUENCE_LENGTH, INPUT_SIZE).to(device)
        batch_y = batch_y.to(device)
        outputs = rnn_model(batch_x)
        _, predict = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predict == batch_y).sum()

    print("Test Accuracy of the model on the {} test images: {}%".format(total, correct * 100 / total))

# Save model
torch.save(rnn_model.state_dict(), "./model/rnn.ckpt")

