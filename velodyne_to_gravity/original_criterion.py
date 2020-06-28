import numpy as np

import torch

def originalCriterion(outputs, labels):
    loss = torch.mean((outputs - labels)**2)
    # print("loss = ", loss)
    loss_mul = torch.mul(loss, 10)
    # print("loss_mul = ", loss_mul)
    return loss_mul

##### test #####
# outputs = np.array([1.1, 2.2, 3.3])
# outputs = torch.from_numpy(outputs)
# print("outputs = ", outputs)
# labels = np.array([2.1, 3.2, 4.3])
# labels = torch.from_numpy(labels)
# print("labels = ", labels)
#
# originalCriterion(outputs, labels)
