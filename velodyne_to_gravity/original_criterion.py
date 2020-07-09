import numpy as np

import torch

def originalCriterion(outputs, labels, a):
    # outputs = torch.mul(outputs, a)
    # labels = torch.mul(labels, a)
    outputs = a*outputs
    labels = a*labels
    loss = torch.mean((outputs - labels)**2)
    return loss

##### test #####
# outputs = np.array([1.1, 2.2, 3.3])
# outputs = torch.from_numpy(outputs)
# print("outputs = ", outputs)
# labels = np.array([2.1, 3.2, 4.3])
# labels = torch.from_numpy(labels)
# print("labels = ", labels)
#
# originalCriterion(outputs, labels)
