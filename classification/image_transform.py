from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms

##### test #####
# import compute_images_mean_std

class image_transform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            "val": transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase="train"):
        # print("image_transform phase: ", phase)
        return self.data_transform[phase](img)

##### test #####
# size = 224  #VGG16
# dir_name = "./data/train/dog"
# file_type = "jpeg"
# mean, std = compute_images_mean_std.compute_images_mean_std(dir_name, file_type)
#
# image_file_path = './data/train/dog/dog_01.jpeg'
# img = Image.open(image_file_path)
#
# transform = image_transform(size, mean, std)
# img_transformed = transform(img, phase="train")
#
# img_transformed = img_transformed.numpy().transpose((1, 2, 0))  #(rgb, h, w) -> (h, w, rgb)
# img_transformed = np.clip(img_transformed, 0, 1)
#
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.subplot(1, 2, 2)
# plt.imshow(img_transformed)
# plt.show()
