import torch.utils.data as data
from PIL import Image

import make_datapath_list
import compute_images_mean_std
import data_transform

class original_dataset(data.Dataset):
    def __init__(self, data_list, transform, phase):
        self.data_list = data_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index][3]
        img = Image.open(img_path)
        img_transformed = self.transform(img, phase=self.phase)

        label = self.data_list[index][:3]

        return img_transformed, label

##### test #####
# rootpath = "/home/amsl/ros_catkin_ws/src/save_dataset/dataset"
# csv_name = "save_image_with_imu.csv"
# train_list = make_datapath_list.make_datapath_list(rootpath, csv_name, phase="train")
# val_list = make_datapath_list.make_datapath_list(rootpath, csv_name, phase="val")
#
# size = 224  #VGG16
# dir_name = "/home/amsl/ros_catkin_ws/src/save_dataset/dataset/train"
# file_type = "jpg"
# mean, std = compute_images_mean_std.compute_images_mean_std(dir_name, file_type, resize=size)
#
# train_dataset = original_dataset(
#     data_list=train_list,
#     transform=data_transform.data_transform(size, mean, std),
#     phase="train"
# )
# val_dataset = original_dataset(
#     data_list=val_list,
#     transform=data_transform.data_transform(size, mean, std),
#     phase="val"
# )
#
# index = 0
# print("index", index, ": ", train_dataset.__getitem__(index)[0].size())   #data
# print("index", index, ": ", train_dataset.__getitem__(index)[1])   #label
