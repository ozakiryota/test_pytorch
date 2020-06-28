import torch.utils.data as data
from PIL import Image
import numpy as np

import torch

import make_datapath_list
import data_transform

class OriginalDataset(data.Dataset):
    def __init__(self, data_list, transform, phase):
        self.data_list = data_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        mat_path = self.data_list[index][4]
        mat = np.load(mat_path)

        acc_str_list = self.data_list[index][:3]
        acc_list = [float(num) for num in acc_str_list]
        acc = np.array(acc_list)

        mat_trans, acc_trans = self.transform(mat, acc, phase=self.phase)

        return mat_trans, acc_trans

##### test #####
# ## list
# rootpath = "/home/amsl/ros_catkin_ws/src/save_dataset/dataset/imu_camera_velodyne"
# csv_name = "imu_color_depth.csv"
# train_list = make_datapath_list.make_datapath_list(rootpath, csv_name, phase="train")
# val_list = make_datapath_list.make_datapath_list(rootpath, csv_name, phase="val")
#
# ## dataset
# train_dataset = OriginalDataset(
#     data_list=train_list,
#     transform=data_transform.data_transform(),
#     phase="train"
# )
# val_dataset = OriginalDataset(
#     data_list=val_list,
#     transform=data_transform.data_transform(),
#     phase="val"
# )
#
# index = 0
# print("index", index, ": ", train_dataset.__getitem__(index)[0].shape)   #data
# print("index", index, ": ", train_dataset.__getitem__(index)[1])   #label
