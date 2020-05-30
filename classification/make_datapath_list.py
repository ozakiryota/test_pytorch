import glob

def make_datapath_list(rootpath, file_type, phase="train"):
    target_path = rootpath + "/" + phase + "/**/*." + file_type
    path_list = glob.glob(target_path)
    return path_list

##### test #####
# rootpath = "./dataset_lion_tiger"
# file_type = "jpg"
# train_list = make_datapath_list(rootpath, file_type, phase="train")
# val_list = make_datapath_list(rootpath, file_type, phase="val")
# print(train_list)
