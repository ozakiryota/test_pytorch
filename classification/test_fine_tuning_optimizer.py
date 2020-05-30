from torchvision import models
import torch.nn as nn
import torch.optim as optim

# network
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

# param
params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []

update_param_names_1  = ["features"]
update_param_names_2  = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
update_param_names_3  = ["classifier.6.weight", "classifier.6.bias"]

for name, param in net.named_parameters():
    # print(name)
    # print(param)

    if update_param_names_1[0] in name:
        param.requires_grad = True
        params_to_update_1.append(param)
        print("add to params_to_update_1: ", name)
    elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
        print("add to params_to_update_2: ", name)
    elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
        print("add to params_to_update_3: ", name)
    else:
        param.requires_grad = False

print("----------")
print("params_to_update_1:\n", params_to_update_1)
print("params_to_update_2:\n", params_to_update_2)
print("params_to_update_3:\n", params_to_update_2)

# optimizer
optimizer = optim.SGD([
    {"params": params_to_update_1, "lr": 1e-4},
    {"params": params_to_update_2, "lr": 5e-4},
    {"params": params_to_update_3, "lr": 1e-3},
], momentum=0.9)

print(optimizer)
