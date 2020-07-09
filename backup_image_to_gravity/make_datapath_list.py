import csv

def make_datapath_list(rootpath, csv_name, phase="train"):
    csvpath = rootpath + "/" + phase + "/" + csv_name
    with open(csvpath) as csvfile:
        reader = csv.reader(csvfile)
        data_list = []
        for row in reader:
            # print(row)
            # print(row[:2])
            # print(row[3])
            data_list.append(row)
    return data_list

##### test #####
# rootpath = "/home/amsl/ros_catkin_ws/src/save_dataset/dataset"
# csv_name = "save_image_with_imu.csv"
# train_list = make_datapath_list(rootpath, csv_name, phase="train")
# val_list = make_datapath_list(rootpath, csv_name, phase="val")
# print(train_list)
# print("example: ", train_list[0][:3], train_list[0][3])
