import copy

import torch
from torch import nn


class ConvolutionalPart(nn.Module):

    def __init__(self):
        super(ConvolutionalPart, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.act4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.act5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        self.flatten1 = nn.Flatten(1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.act4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.act5(x)
        x = self.pool5(x)

        x = self.flatten1(x)

        return x


class ModelStage1(nn.Module):

    def __init__(self, cnn_part):
        super(ModelStage1, self).__init__()
        self.cnn_part = cnn_part
        self.fc1 = nn.Linear(256 * 3 * 3, 256 * 3 * 3)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(256 * 3 * 3, 8)
        self.sgm = nn.Sigmoid()

    def forward(self, x):
        x = self.cnn_part(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.sgm(x)

        return x


class ModelStage2(nn.Module):

    def __init__(self, stage_1_model):
        super(ModelStage2, self).__init__()

        self.stage_1_model = stage_1_model

        self.cnn_part = ConvolutionalPart()
        self.cnn_part.load_state_dict(copy.deepcopy(stage_1_model.cnn_part.state_dict()))

        self.fc1 = nn.Linear(256 * 3 * 3 + 8, 256 * 3 * 3)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(256 * 3 * 3, 22)
        self.sgm = nn.Sigmoid()

    def forward(self, x):
        self.stage_1_model.eval()
        with torch.no_grad():
            stage_1_res = self.stage_1_model(x)

        x = self.cnn_part(x)

        # add stage 1 result as features

        new_x = torch.cat((stage_1_res, x), 1)

        x = self.fc1(new_x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.sgm(x)

        return torch.cat((stage_1_res, x), 1)