from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import make_datapath_list
import data_transform
import original_dataset
import original_network

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device = ", device)

    net.to(device)

    # loss record
    writer = SummaryWriter()
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

            if (epoch == 0) and (phase=="train"):
                continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                ## initialize optimizer
                optimizer.zero_grad()   #reset grad to zero (after .step())

                with torch.set_grad_enabled(phase == "train"):  #compute grad only in "train"
                    ## forward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    ## backward
                    if phase == "train":
                        loss.backward()     #accumulate gradient to each Tensor
                        optimizer.step()    #update param depending on current .grad

                    epoch_loss += loss.item() * inputs.size(0)
                    # print("loss.item() = ", loss.item())

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            if phase == "train":
                record_loss_train.append(epoch_loss)
                writer.add_scalar("Loss/train", epoch_loss, epoch)
            else:
                record_loss_val.append(epoch_loss)
                writer.add_scalar("Loss/val", epoch_loss, epoch)
    ## save param
    save_path = "./weights/weights_image_to_gravity.pth"
    torch.save(net.state_dict(), save_path)
    print("Parameter file is saved as ", save_path)

    ## graph
    graph = plt.figure()
    plt.plot(range(len(record_loss_train)), record_loss_train, label="Train")
    plt.plot(range(len(record_loss_val)), record_loss_val, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    graph.savefig("./graph/graph.jpg")
    plt.show()

    writer.close()

##### execution #####
## random
keep_reproducibility = False
if keep_reproducibility:
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

## list
train_rootpath = "/home/amsl/ozaki/airsim_ws/pkgs/airsim_controller/save/train"
val_rootpath = "/home/amsl/ozaki/airsim_ws/pkgs/airsim_controller/save/val"
csv_name = "imu_camera.csv"
train_list = make_datapath_list.make_datapath_list(train_rootpath, csv_name)
val_list = make_datapath_list.make_datapath_list(val_rootpath, csv_name)

## trans param
size = 224  #VGG16
mean = ([0.25, 0.25, 0.25])
std = ([0.5, 0.5, 0.5])

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
batch_size = 100
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
print("train data: ", len(dataloaders_dict["train"].dataset))
print("val data: ", len(dataloaders_dict["val"].dataset))

## criterion
criterion = nn.MSELoss()

## network
net = original_network.OriginalNet()
print(net)
# vgg = models.vgg16(pretrained=True)
# print(vgg)


## param
list_cnn_param_value, list_fc_param_value = net.getParamValueList()

## optimizer
optimizer = optim.SGD([
    {"params": list_cnn_param_value, "lr": 1e-4},
    {"params": list_fc_param_value, "lr": 1e-3}
], momentum=0.9)
print(optimizer)

## execution
num_epochs = 200
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
