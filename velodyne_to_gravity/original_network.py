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
        return x

##### test #####
# net = OriginalNet()
# print(net)
# for param_name, param_value in net.named_parameters():
#     print(param_name)
# image_path = "/home/amsl/ros_catkin_ws/src/save_dataset/dataset/imu_camera_velodyne/depth_example.jpg"
# img = Image.open(image_path)
# acc = np.array([0, 0, 1])
# transform = data_transform.data_transform()
# img_transformed, _ = transform(img, acc, phase="train")
# inputs = img_transformed.unsqueeze_(0)
# outputs = net(inputs)
# print("img_transformed.size() = ", img_transformed.size())
# print("inputs.size() = ", inputs.size())
# print("outputs.size() = ", outputs.size())
