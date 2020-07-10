import histomicstk as htk
import glob
import cv2
import os
import numpy as np
import scipy as sp

import skimage.io
import skimage.measure
import skimage.color


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
%matplotlib inline


def reinhard_img(directory, des_path):
    img = cv2.imread('Mix_crypts/train/Images/1005475 (1, 16160, 31023, 1457, 443).png')
    mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(img)

    for filename in os.listdir(directory):
        print(filename)
        if filename.endswith(".png"):
            img_input = cv2.imread(directory + filename)
            plt.imshow(img_input), plt.show()
            im_nmzd = htk.preprocessing.color_normalization.reinhard(img_input, mean_ref, std_ref)
            status = cv2.imwrite(des_path + filename, im_nmzd)
            plt.imshow(im_nmzd), plt.show()
            print("Image written to file-system : ", status)
            continue
        else:
            continue


reinhard_img('Mix_crypts/valid/Images/', 'Reinhard_Mix_crypts/valid/Images/')