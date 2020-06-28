import csv
import os

def make_datapath_list(rootpath, csv_name, phase="train"):
    csvpath = os.path.join(rootpath, phase, csv_name)
    with open(csvpath) as csvfile:
        reader = csv.reader(csvfile)
        data_list = []
        for row in reader:
            # print(row)
            # print(row[:3])
            # print(row[3])
            row[3] = os.path.join(rootpath, phase, row[3])
            row[4] = os.path.join(rootpath, phase, row[4])
            data_list.append(row)
    return data_list

##### test #####
# rootpath = "/home/amsl/ros_catkin_ws/src/save_dataset/dataset/imu_camera_velodyne"
# csv_name = "imu_color_depth.csv"
# train_list = make_datapath_list(rootpath, csv_name, phase="train")
# val_list = make_datapath_list(rootpath, csv_name, phase="val")
# # print(train_list)
# print("example: ", train_list[0][:3], train_list[0][3], train_list[0][4])
