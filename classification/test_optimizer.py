from torchvision import models
import torch.nn as nn
import torch.optim as optim

# network
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

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

print(optimizer)
