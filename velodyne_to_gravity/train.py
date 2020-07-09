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
import original_criterion

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
                    # loss = criterion(outputs, labels)
                    loss = original_criterion.originalCriterion(outputs, labels, 30)

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
                writer.add_scalar("Loss/train", epoch_loss, epoch)
            else:
                record_loss_val.append(epoch_loss)
                writer.add_scalar("Loss/val", epoch_loss, epoch)
    # save param
    save_path = "./weights/weights_velodyne_to_gravity.pth"
    torch.save(net.state_dict(), save_path)
    print("Parameter file is saved as ", save_path)

    # graph
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
# rootpath = "/home/amsl/ros_catkin_ws/src/save_dataset/dataset/imu_camera_velodyne"
train_rootpath = "/home/amsl/ros_catkin_ws/src/save_dataset/dataset/imu_camera_velodyne/2019-01-13-15-46-58_filtered_normalized"
val_rootpath = "/home/amsl/ros_catkin_ws/src/save_dataset/dataset/imu_camera_velodyne/2019-12-07_filtered_normalized"
csv_name = "imu_color_depth.csv"
train_list = make_datapath_list.make_datapath_list(train_rootpath, csv_name)
val_list = make_datapath_list.make_datapath_list(val_rootpath, csv_name)

## dataset
train_dataset = original_dataset.OriginalDataset(
    data_list=train_list,
    transform=data_transform.data_transform(),
    phase="train"
)
val_dataset = original_dataset.OriginalDataset(
    data_list=val_list,
    transform=data_transform.data_transform(),
    phase="val"
)

## dataloader
batch_size = 50
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

## optimizer
optimizer = optim.SGD(params=net.parameters(), lr=1e-4, momentum=0.9)
print(optimizer)

## execution
num_epochs = 30
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
