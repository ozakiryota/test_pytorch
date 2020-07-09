from PIL import Image
import numpy as np

import torch
from torchvision import models
import torch.nn as nn

import data_transform

class OriginalNet(nn.Module):
    def __init__(self):
        super(OriginalNet, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(17152, 18),
            nn.ReLU(),
            nn.Linear(18, 3),
        )

    def forward(self, x):
        # print("cnn-in", x.size())
        x = self.cnn(x)
        # print("cnn-out", x.size())
        x = torch.flatten(x, 1)
        # print("fc-in", x.size())
        x = self.fc(x)
        # print("fc-out", x.size())
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

##### test #####
# net = OriginalNet()
# print(net)
# for param_name, param_value in net.named_parameters():
#     print(param_name)
# mat_file_path = "/home/amsl/ros_catkin_ws/src/save_dataset/dataset/imu_camera_velodyne/example_depth.npy"
# mat = np.load(mat_file_path)
# acc = np.array([0, 0, 1])
# transform = data_transform.data_transform()
# mat_trans, _ = transform(mat, acc, phase="train")
# print("mat_trans.size() = ", mat_trans.size())
# inputs = mat_trans.unsqueeze_(0)
# print("inputs.size() = ", inputs.size())
# outputs = net(inputs)
# print("outputs.size() = ", outputs.size())
