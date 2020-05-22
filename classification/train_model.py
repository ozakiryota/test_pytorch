from tqdm import tqdm

import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim

import make_datapath_list
import compute_images_mean_std
import image_transform
import original_dataset

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("----------")

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase=="train"):
                continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                # initialize optimizer
                optimizer.zero_grad()   #reset grad to zero (after .step())

                with torch.set_grad_enabled(phase == "train"):  #compute grad only in "train"
                    # forward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)  #return (values, indices)

                    # backward
                    if phase == "train":
                        loss.backward()     #accumulate gradient to each Tensor
                        optimizer.step()    #update pram depending on current .grad

                    epoch_loss += loss.item() * inputs.size(0)  #loss.item(): average loss in batch, inputs.size(0): 32 images
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
    # save param
    save_path = "./weights/weights_lion_tiger.pth"
    torch.save(net.state_dict(), save_path)
    print("Parameter file is saved as ", save_path)

##### execution #####
# network
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
print(net)

# file path list
rootpath = "./dataset_lion_tiger"
file_type = "jpg"
train_list = make_datapath_list.make_datapath_list(rootpath, file_type, phase="train")
val_list = make_datapath_list.make_datapath_list(rootpath, file_type, phase="val")

# mean, std
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

# criterion
criterion = nn.CrossEntropyLoss()

# param
params_to_update = []
update_param_names  = ["classifier.6.weight", "classifier.6.bias"]
for name, param in net.named_parameters():
    # print(name)
    # print(param)
    if name in update_param_names:
        print(name)
        param.requires_grad = True
        params_to_update.append(param)
    else:
        param.requires_grad = False
print("----------")
print(params_to_update)

# optimizer
optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)  #lr: learning rate

# execution
num_epochs = 2
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
