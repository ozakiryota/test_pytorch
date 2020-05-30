from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision import models
import torch.nn as nn

import make_datapath_list
import compute_images_mean_std
import image_transform

## network
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
net.eval()

# load_path = "./weights/cpu/weights_lion_tiger.pth"
load_path = "./weights/gpu/weights_lion_tiger_finetuning.pth"

## saved in CPU -> load in CPU, saved in GPU -> load in GPU
load_weights = torch.load(load_path)

## saved in GPU -> load in CPU
# load_weights = torch.load(load_path, map_location={"cuda:0": "cpu"})

net.load_state_dict(load_weights)

## input
img_rootpath = "./dataset_lion_tiger"
file_type = "jpg"
val_list = make_datapath_list.make_datapath_list(img_rootpath, file_type, phase="val")

size = 224  #VGG16
mean, std = compute_images_mean_std.compute_images_mean_std("./dataset_lion_tiger/train", "jpg", resize=size)
transform = image_transform.image_transform(size, mean, std)

## predict(0: lion, 1:tiger)
plt.figure()
i = 0
h = 4
w = 5

for img_path in val_list:
    img = Image.open(img_path)
    img_trasformed = transform(img)
    inputs = img_trasformed.unsqueeze_(0)
    outputs = net(inputs)
    predict_id = np.argmax(outputs.detach().numpy())
    predict_name = ""
    print(outputs)
    if predict_id == 0:
        predict_name = "lion"
    else:
        predict_name = "tiger"

    print("predict: ", predict_name)
    plt.subplot(h, w, i+1)
    plt.title(predict_name)
    plt.imshow(img)

    i += 1

plt.show()
