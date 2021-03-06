from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import math

from torchvision import transforms

import compute_images_mean_std

class data_transform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            "val": transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, acc, phase="train"):
        if phase == "train":
        # if (phase == "train") or (phase == "val"):
            angle_deg = random.uniform(-45, 45)
            angle_rad = angle_deg / 180 * math.pi
            # print("angle_deg = ", angle_deg)

            # vector rotation
            rot = np.array([
                [1, 0, 0],
                [0, math.cos(-angle_rad), -math.sin(-angle_rad)],
                [0, math.sin(-angle_rad), math.cos(-angle_rad)]
            ])
            acc = np.dot(rot, acc)

            # image rotation
            img = img.rotate(angle_deg)
            img = self.data_transform[phase](img)
        else:
            img = self.data_transform[phase](img)

        acc = acc.astype(np.float32)
        return img, acc

##### test #####
# size = 224  #VGG16
# dir_name = "/home/amsl/ros_catkin_ws/src/save_dataset/dataset/train"
# file_type = "jpg"
# mean, std = compute_images_mean_std.compute_images_mean_std(dir_name, file_type, resize=size)
#
# # image_file_path = "/home/amsl/ros_catkin_ws/src/save_dataset/dataset/example.jpg"
# image_file_path = "../classification/dataset_lion_tiger/lion_example.jpg"
# img = Image.open(image_file_path)
# print("img.size = ", img.size)
#
# g_list = [0, 0, 1]
# acc = np.array(g_list)
#
# transform = data_transform(size, mean, std)
# img_transformed, acc_transformed = transform(img, acc, phase="train")
# print("acc_transformed", acc_transformed)
#
# img_transformed = img_transformed.numpy().transpose((1, 2, 0))  #(rgb, h, w) -> (h, w, rgb)
# img_transformed = np.clip(img_transformed, 0, 1)
# print("img_transformed.shape = ", img_transformed.shape)
#
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.subplot(1, 2, 2)
# plt.imshow(img_transformed)
# plt.show()
