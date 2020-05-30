import numpy as np
import glob
from PIL import Image

def compute_images_mean_std(rootpath, file_type, resize=-1):
    file_list = glob.glob(rootpath + "/**/*." + file_type)

    data_appended = np.empty([0, 3])
    for path in file_list:
        # print(path)
        img = Image.open(path)
        if resize != -1:
            w, h = img.size
            ratio = resize / w
            img = img.resize((int(w * ratio), int(h * ratio)))
            # print("shrink: ", int(w * ratio), "x", int(h * ratio))
        # print("np.asarray(img) = \n", np.asarray(img))
        data = np.asarray(img)/255
        data_reshaped = data.reshape(-1, 3)
        data_appended = np.append(data_appended, data_reshaped, axis=0)
    mean = data_appended.mean(axis=0)
    std = data_appended.std(axis=0)

    print("data_appended = \n", data_appended)
    print("data_appended.shape = ", data_appended.shape)
    print("mean = ", mean)
    print("std = ", std)
    
    return mean, std

##### test #####
# rootpath = "./ILSVRC2012/hymenoptera_data/train/ants"
# file_type = "jpg"
# mean, std = compute_images_mean_std(rootpath, file_type)
