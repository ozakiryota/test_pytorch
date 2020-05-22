import torch.utils.data as data
from PIL import Image

##### test #####
# import make_datapath_list
# import compute_images_mean_std
# import image_transform

class original_dataset(data.Dataset):
    def __init__(self, file_list, transform, phase):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(img, phase=self.phase)

        label_string = img_path.split("/")[3]
        label = -1
        if label_string == "lion":
            label = 0
        if label_string == "tiger":
            label = 1
        # print("index = ", index, "   label: ", label_string, " -> ", label)

        return img_transformed, label

##### test #####
# rootpath = "./data"
# file_type = "jpeg"
# train_list = make_datapath_list.make_datapath_list(rootpath, file_type, phase="train")
# val_list = make_datapath_list.make_datapath_list(rootpath, file_type, phase="val")
#
# size = 224  #VGG16
# mean, std = compute_images_mean_std.compute_images_mean_std("./data/train", "jpeg")
#
# train_dataset = original_dataset(
#     file_list=train_list,
#     transform=image_transform.image_transform(size, mean, std),
#     phase="train"
# )
# val_dataset = original_dataset(
#     file_list=val_list,
#     transform=image_transform.image_transform(size, mean, std),
#     phase="val"
# )
#
# index = 0
# print("index", index, ": ", train_dataset.__getitem__(index)[0].size())   #data
# print("index", index, ": ", train_dataset.__getitem__(index)[1])   #label
