from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

import torch
from torchvision import models
import torch.nn as nn

import compute_images_mean_std
import make_datapath_list
import data_transform
import original_dataset

## device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = ", device)

## network
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.features[26] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
net.features = nn.Sequential(*list(net.features.children())[:-3])
net.classifier = nn.Sequential(
    nn.Linear(in_features=73728, out_features=18, bias=True),
    nn.ReLU(True),
    nn.Linear(in_features=18, out_features=3, bias=True)
)
print(net)
net.to(device)
net.eval()

## saved in CPU -> load in CPU, saved in GPU -> load in GPU
load_path = "./weights/weights_image_to_gravity.pth"
load_weights = torch.load(load_path)
net.load_state_dict(load_weights)

## mean, std
size = 224  #VGG16
# dir_name = "/home/amsl/ros_catkin_ws/src/save_dataset/dataset/train"
# file_type = "jpg"
# mean, std = compute_images_mean_std.compute_images_mean_std(dir_name, file_type, resize=size)
mean = ([0.5, 0.5, 0.5])
std = ([0.25, 0.25, 0.25])

## list
rootpath = "/home/amsl/ros_catkin_ws/src/save_dataset/dataset"
csv_name = "save_image_with_imu.csv"
train_list = make_datapath_list.make_datapath_list(rootpath, csv_name, phase="train")
val_list = make_datapath_list.make_datapath_list(rootpath, csv_name, phase="val")

## transform
transform = data_transform.data_transform(size, mean, std)

## dataset
train_dataset = original_dataset.OriginalDataset(
    data_list=train_list,
    transform=data_transform.data_transform(size, mean, std),
    phase="train"
)
val_dataset = original_dataset.OriginalDataset(
    data_list=val_list,
    transform=data_transform.data_transform(size, mean, std),
    phase="val"
)

## dataloader
batch_size = 50
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

## predict
# batch_iterator = iter(dataloaders_dict["train"])
batch_iterator = iter(dataloaders_dict["val"])
inputs, labels = next(batch_iterator)
inputs_device = inputs.to(device)
labels_device = labels.to(device)
outputs = net(inputs_device)
print("tesy")

plt.figure()
i = 0
h = 5
w = 10

sum_r = 0.0
sum_p = 0.0
def accToRP(acc):
    r = math.atan2(acc[1], acc[2])
    p = math.atan2(-acc[0], math.sqrt(acc[1]*acc[1] + acc[2]*acc[2]))
    print("r[deg]: ", r/math.pi*180.0, " p[deg]: ", p/math.pi*180.0)
    return r, p

for i in range(inputs.size(0)):
    print(i)
    print("label: ", labels[i])
    print("output: ", outputs[i])
    l_r, l_p = accToRP(labels[i])
    o_r, o_p = accToRP(outputs[i])
    e_r = math.atan2(math.sin(l_r - o_r), math.cos(l_r - o_r))
    e_p = math.atan2(math.sin(l_p - o_p), math.cos(l_p - o_p))
    print("e_r[deg]: ", e_r/math.pi*180.0, " e_p[deg]: ", e_p/math.pi*180.0)
    sum_r += abs(e_r)
    sum_p += abs(e_p)
    if i < h*w:
        plt.subplot(h, w, i+1)
        plt.imshow(np.clip(inputs[i].numpy().transpose((1, 2, 0)), 0, 1))
        plt.title(i)
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

print("---ave---\n e_r[deg]: ", sum_r/inputs.size(0)/math.pi*180.0, " e_p[deg]: ", sum_p/inputs.size(0)/math.pi*180.0)

plt.show()
