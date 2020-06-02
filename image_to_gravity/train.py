from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim

import make_datapath_list
import compute_images_mean_std
import data_transform
import original_dataset

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device = ", device)

    net.to(device)

    torch.backends.cudnn.benchmark = True

    # loss record
    record_loss_train = []
    record_loss_val = []

    for epoch in range(num_epochs):
        print("----------")
        print("Epoch {}/{}".format(epoch+1, num_epochs))

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0

            # if (epoch == 0) and (phase=="train"):
            #     continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # initialize optimizer
                optimizer.zero_grad()   #reset grad to zero (after .step())

                with torch.set_grad_enabled(phase == "train"):  #compute grad only in "train"
                    # forward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    # backward
                    if phase == "train":
                        loss.backward()     #accumulate gradient to each Tensor
                        optimizer.step()    #update param depending on current .grad

                    epoch_loss += loss.item() * inputs.size(0)  #loss.item(): average loss in batch, inputs.size(0): 32 images
                    # print("loss.item() = ", loss.item())
                    # print("inputs.size(0) = ", inputs.size(0))

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            if phase == "train":
                record_loss_train.append(epoch_loss)
            else:
                record_loss_val.append(epoch_loss)
    # save param
    save_path = "./weights/weights_image_to_gravity.pth"
    torch.save(net.state_dict(), save_path)
    print("Parameter file is saved as ", save_path)

    # graph
    plt.plot(range(len(record_loss_train)), record_loss_train, label="Train")
    plt.plot(range(len(record_loss_val)), record_loss_val, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.show()

##### execution #####
## random
keep_reproducibility = False
if keep_reproducibility:
    # CPU
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

## list
rootpath = "/home/amsl/ros_catkin_ws/src/save_dataset/dataset"
csv_name = "save_image_with_imu.csv"
train_list = make_datapath_list.make_datapath_list(rootpath, csv_name, phase="train")
val_list = make_datapath_list.make_datapath_list(rootpath, csv_name, phase="val")

## mean, std
size = 224  #VGG16
file_type = "jpg"
mean, std = compute_images_mean_std.compute_images_mean_std(rootpath + "/train", file_type, resize=size)

## dataset
train_dataset = original_dataset.OriginalDataset(
    data_list=train_list,
    transform=data_transform.data_transform(size, mean, std),
    phase="train"
)
val_dataset = original_dataset.OriginalDataset(
    data_list=val_list,
    transform=data_transform.data_transform(size, mean, std),
    phase="val"
)

## dataloader
batch_size = 30
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

## criterion
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

## network
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.features[26] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
net.features = nn.Sequential(*list(net.features.children())[:-3])
net.classifier = nn.Sequential(
    nn.Linear(in_features=73728, out_features=18, bias=True),
    nn.ReLU(True),
    nn.Linear(in_features=18, out_features=3, bias=True),
    nn.ReLU(True)
)
print(net)

## param
params_to_update_1 = []
params_to_update_2 = []

update_param_names_1  = ["features.26.weight", "features.26.bias"]
update_param_names_2  = ["classifier"]

for name, param in net.named_parameters():
    # print(name)
    # print(param)

    if name in update_param_names_1:
        param.requires_grad = True
        params_to_update_1.append(param)
        print("add to params_to_update_1: ", name)
    elif update_param_names_2[0] in name:
        param.requires_grad = True
        params_to_update_2.append(param)
        print("add to params_to_update_2: ", name)
    else:
        param.requires_grad = False

print("----------")
# print("params_to_update_1:\n", params_to_update_1)
# print("params_to_update_2:\n", params_to_update_2)

## optimizer
optimizer = optim.SGD([
    {"params": params_to_update_1, "lr": 5e-6},
    {"params": params_to_update_2, "lr": 5e-6}
], momentum=0.9)
print(optimizer)

## execution
num_epochs = 20
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
