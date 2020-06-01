import numpy as np
import matplotlib.pyplot as plt

import torch

import make_datapath_list
import compute_images_mean_std
import image_transform
import original_dataset

def show_inputs(inputs):
    h = 4
    w = 8
    i = 0
    plt.figure()
    for tensor in inputs:
        # print(tensor.size())
        img = tensor.numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1)
        plt.subplot(h, w, i+1)
        plt.imshow(img)
        i += 1
    plt.show()

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
# batch_iterator = iter(dataloaders_dict["val"])
inputs, labels = next(batch_iterator)

print(inputs.size())
print(labels)
show_inputs(inputs)
