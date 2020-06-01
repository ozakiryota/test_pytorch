from PIL import Image

from torchvision import models
import torch.nn as nn

import data_transform

use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
print("vgg16: \n", net)

net.features[26] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
new_features = nn.Sequential(*list(net.features.children())[:-3])
net.features = new_features

net.classifier = nn.Sequential(
    nn.Linear(in_features=73728, out_features=18, bias=True),
    nn.ReLU(True),
    nn.Linear(in_features=18, out_features=3, bias=True),
    nn.ReLU(True)
)

print("original network: \n", net)

image_path = "/home/amsl/ros_catkin_ws/src/save_dataset/dataset/train/img0.jpg"
img = Image.open(image_path)
transform = data_transform.data_transform(224, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
img_transformed = transform(img, phase="train")
inputs = img_transformed.unsqueeze_(0)
outputs = net(inputs)
print("img_transformed.size() = ", img_transformed.size())
print("inputs.size() = ", inputs.size())
print("outputs.size() = ", outputs.size())
