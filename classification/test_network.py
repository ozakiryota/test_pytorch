from torchvision import models
import torch.nn as nn

use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
print("vgg16: \n", net)

net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

net.train() #train mode

print("original network: \n", net)
print("Completed network config")
