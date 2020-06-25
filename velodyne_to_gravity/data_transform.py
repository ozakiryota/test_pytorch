from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import math

from torchvision import transforms

class data_transform():
    def __init__(self):
        self.data_transform = {
            "train": transforms.Compose([
                transforms.ToTensor(),
            ]),
            "val": transforms.Compose([
                transforms.ToTensor(),
            ])
        }

    def __call__(self, img, acc, phase="train"):
        img = self.data_transform[phase](img)
        acc = acc.astype(np.float32)
        return img, acc

##### test #####
# image_file_path = "/home/amsl/ros_catkin_ws/src/save_dataset/dataset/imu_camera_velodyne/depth_example.jpg"
# img = Image.open(image_file_path)
# print("img.size = ", img.size)
#
# g_list = [0, 0, 1]
# acc = np.array(g_list)
#
# transform = data_transform()
# img_transformed, acc_transformed = transform(img, acc, phase="train")
# print("img_transformed.shape = ", img_transformed.shape)
# print("img_transformed = ", img_transformed)
# print("acc_transformed = ", acc_transformed)
#
# img_transformed_numpy = img_transformed.numpy().transpose((1, 2, 0))  #(rgb, h, w) -> (h, w, rgb)
# img_transformed_numpy = img_transformed_numpy.squeeze(2)
# print("img_transformed_numpy.shape = ", img_transformed_numpy.shape)
#
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.imshow(img)
# plt.subplot(2, 1, 2)
# plt.imshow(img_transformed_numpy)
# plt.show()
