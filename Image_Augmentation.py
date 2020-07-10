#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import imgaug.augmenters as iaa
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import imutils
import random


# In[ ]:


# for each image:
# rotation zoom blur
# rotation:4 
# zoom: 2 (in and out)
# blur: 1
# rotation+zoom: 3


# In[ ]:





# In[ ]:


# rotate+zoom in
def method2(images):
    seq = iaa.Sequential([
        iaa.Flipud(0.5),
        iaa.Sometimes(0.5,iaa.Affine(scale = (0.7, 0.9),rotate=(-180,180))),
    ], random_order=True)
    
    images_aug = seq(images=images)
    return images_aug


# In[ ]:


# rotate + zoom in
def method3(images):
    seq = iaa.Sequential([
        #iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Sometimes(0.6,iaa.Affine(scale = (0.5, 0.7),rotate=(-180,180))),
    ], random_order=True)
    
    images_aug = seq(images=images)
    return images_aug


# In[ ]:


# rotate + zoom in + blur
def method4(images):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Sometimes(0.6,iaa.Affine(scale = (0.8, 0.9),rotate=(-180,180))),
        iaa.GaussianBlur(sigma=(0, 3.0))
    ], random_order=True)
    
    images_aug = seq(images=images)
    return images_aug


# In[ ]:


imgs = []
directory = 'color_normalized_train_imgs/'
for filename in os.listdir(directory):
    if filename.endswith(".png"): 
        imgs.append(cv2.imread(directory+filename))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# rotate random -180 to 180, 5 versions 
def rotate5(filename):
    image = cv2.imread(filename) 
    rot_imgs = []
    for i in np.arange(0,5):
        angle = random.randint(-180,180)
        rotated = imutils.rotate_bound(image, angle)
        rot_imgs.append(rotated)
    #print(len(rot_imgs))
    return rot_imgs


# In[ ]:


# zoom in 
def zoom3(rot_imgs): 
    seq = iaa.Sequential([
        iaa.Sometimes(0.5,iaa.Affine(scale = (0.7, 0.9)))
        ], random_order=True)

    zoomed = seq(images=rot_imgs[:3])
    plt.imshow(zoomed[0]), plt.show()
    #print(len(zoomed))
    return zoomed


# In[ ]:


# rotate + zoom in + blur
def blur1(rot_imgs):
    seq = iaa.Sequential([
        angle = random.randint(-180,180)
        rotated = imutils.rotate_bound(image, angle)
        iaa.Sometimes(0.6,iaa.Affine(scale = (0.8, 0.9))),
        iaa.GaussianBlur(sigma=(0, 3.0))
    ], random_order=True)
    
    blurred = seq(images=rot_imgs[3:])
    plt.imshow(blurred[0]), plt.show()
    print(len(blurred))
    return blurred


# In[ ]:


imgs = []
directory = 'color_normalized_train_imgs/'
results = []
for filename in os.listdir(directory):
    if filename.endswith(".png"): 
        curr_img = cv2.imread(directory+filename)
        rot_imgs = rotate5(curr_img)
        zoomed = zoom3(rot_imgs)
        blurred = blur1(rot_imgs)
        for i in range(len(zoomed)):
            results.append(zoomed[i])
        for i in range(len(blurred)):
            results.append(blurred[i])
        print(len(results))


# In[ ]:


for i in results:
    status = cv2.imwrite('Image_Augmentation/'+'Image'+str(i)+'.png', I)
    print("Image written to file-system : ",status)


# In[ ]:


cnt = 0
for filename in os.listdir(directory):
    cnt+=1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




