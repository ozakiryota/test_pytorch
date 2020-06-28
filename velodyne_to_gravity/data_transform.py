from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

import torch
from torchvision import transforms

class data_transform():
    def __init__(self):
        self.data_transform = {
            "train": transforms.Compose([
            ]),
            "val": transforms.Compose([
            ])
        }

    def __call__(self, mat, acc, phase="train"):
        # mat = self.data_transform[phase](mat)
        mat = mat.astype(np.float32)
        mat_tensor = torch.from_numpy(mat)
        mat_tensor = mat_tensor.unsqueeze_(0)
        acc = acc.astype(np.float32)
        # acc = acc/(math.sqrt(acc[0]*acc[0] + acc[1]*acc[1] + acc[2]*acc[2]))
        acc_tensor = torch.from_numpy(acc)
        return mat_tensor, acc_tensor

##### test #####
# mat_file_path = "/home/amsl/ros_catkin_ws/src/save_dataset/dataset/imu_camera_velodyne/example_depth.npy"
# mat = np.load(mat_file_path)
# print("mat.shape = ", mat.shape)
# print("mat = ", mat)
#
# g_list = [0, 0, 1]
# acc = np.array(g_list)
#
# transform = data_transform()
# mat_trans, acc_trans = transform(mat, acc, phase="train")
# print("mat_trans.size() = ", mat_trans.size())
# print("mat_trans = ", mat_trans)
# print("acc_trans = ", acc_trans)
#
# img = mat_trans.numpy().transpose((1, 2, 0))  #(ch, h, w) -> (h, w, ch)
# img = img.squeeze(2)
#
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.imshow(mat)
# plt.subplot(2, 1, 2)
# plt.imshow(img)
# plt.show()
