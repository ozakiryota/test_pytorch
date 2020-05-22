import torch

import make_datapath_list
import compute_images_mean_std
import image_transform
import original_dataset

rootpath = "./dataset_lion_tiger"
file_type = "jpg"
train_list = make_datapath_list.make_datapath_list(rootpath, file_type, phase="train")
val_list = make_datapath_list.make_datapath_list(rootpath, file_type, phase="val")

size = 224  #VGG16
mean, std = compute_images_mean_std.compute_images_mean_std("./dataset_lion_tiger/train", "jpg", resize=size)

# dataset
train_dataset = original_dataset.original_dataset(
    file_list=train_list,
    transform=image_transform.image_transform(size, mean, std),
    phase="train"
)
val_dataset = original_dataset.original_dataset(
    file_list=val_list,
    transform=image_transform.image_transform(size, mean, std),
    phase="val"
)

# dataloader
batch_size = 32

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

batch_iterator = iter(dataloaders_dict["train"])
inputs, labels = next(batch_iterator)
print(inputs.size())
print(labels)
