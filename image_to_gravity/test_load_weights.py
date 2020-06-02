from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision import models
import torch.nn as nn

import make_datapath_list
import compute_images_mean_std
import data_transform

## network
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.features[26] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
net.features = nn.Sequential(*list(net.features.children())[:-3])
net.classifier = nn.Sequential(
    nn.Linear(in_features=73728, out_features=18, bias=True),
    nn.ReLU(True),
    nn.Linear(in_features=18, out_features=3, bias=True),
    nn.ReLU(True)
)
print(net)

## saved in CPU -> load in CPU, saved in GPU -> load in GPU
load_path = "./weights/weights_image_to_gravity.pth"
load_weights = torch.load(load_path)
net.load_state_dict(load_weights)

## list
rootpath = "/home/amsl/ros_catkin_ws/src/save_dataset/dataset"
csv_name = "save_image_with_imu.csv"
train_list = make_datapath_list.make_datapath_list(rootpath, csv_name, phase="train")
val_list = make_datapath_list.make_datapath_list(rootpath, csv_name, phase="val")

## mean, std
size = 224  #VGG16
dir_name = "/home/amsl/ros_catkin_ws/src/save_dataset/dataset/train"
file_type = "jpg"
mean, std = compute_images_mean_std.compute_images_mean_std(dir_name, file_type, resize=size)

## transform
transform = data_transform.data_transform(size, mean, std)

plt.figure()
i = 0
h = 10
w = 10

for data in val_list:
    img_path = data[3]
    img = Image.open(img_path)
    img_trasformed = transform(img)

    plt.subplot(h, w, i+1)
    plt.imshow(np.clip(img_trasformed.numpy().transpose((1, 2, 0)), 0, 1))
    plt.title(i)
    # plt.imshow(img)
    plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    inputs = img_trasformed.unsqueeze_(0)
    outputs = net(inputs)
    print(i)
    print("label: ", data[:3])
    print("outputs", outputs)

    i += 1

plt.show()
