#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import imgaug.augmenters as iaa
import torchvision
import skimage as sk
from skimage import io
from skimage import transform
import random
from numpy import ndarray
import imageio
from imgaug import augmenters as iaa
import glob
from PIL import Image
import os
import sys
from tqdm import tqdm
from scipy import ndimage, misc
from skimage import exposure
import cv2
import ntpath


# In[ ]:


def random_rotation(image_array: ndarray, mask1, mask2=None):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-50, -90)
    return sk.transform.rotate(image_array, random_degree, preserve_range=True).astype(np.uint8), sk.transform.rotate(mask, random_degree, preserve_range=True).astype(np.uint8), sk.transform.rotate(mask2, random_degree, preserve_range=True).astype(np.uint8)

def rotate_image(mat: ndarray):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """    
    angle = random.uniform(-25, -50)
    
    height, width = mat.shape[:2] # image shape has 3 dimensions
    
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,0])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    
    return rotated_mat, angle

def rotate_mask(mat: ndarray, ang):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """    
    angle = ang
    
    height, width = mat.shape[:2] # image shape has 3 dimensions
    
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,0])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    
    return rotated_mat

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def randomCrop(img, width=256, height=256):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    mask = mask[y:y+height, x:x+width]
    return img, mask

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

def vertical_flip(image_array: ndarray):
    return image_array[::-1, :]

def blur_image(image_array: ndarray):
    return ndimage.uniform_filter(image_array)

def log_correction(image_array: ndarray):
    return exposure.adjust_log(image_array)

def gamma_correction(image_array: ndarray):
    return exposure.adjust_gamma(image_array, gamma=0.4, gain=0.9)

def sigmoid_correction(image_array: ndarray):
    return exposure.adjust_sigmoid(image_array)

def random_zoom(image_array: ndarray):
    return ndimage.zoom(image_array, 5, order=0)

def distort_affine_skimage(image, rotation=10.0, shear=5.0, random_state=None):
    # distorted affine transformation of the image
    if random_state is None:
        random_state = np.random.RandomState(None)

    rot = np.deg2rad(np.random.uniform(-rotation, rotation))
    sheer = np.deg2rad(np.random.uniform(-shear, shear))
    
    shape1 = image.shape
    shape_size1 = shape1[:2]
    center1 = np.float32(shape_size1) / 2. - 0.5

    pre1 = transform.SimilarityTransform(translation=-center1)
    affine1 = transform.AffineTransform(rotation=rot, shear=sheer, translation=center1)
    #flip1 = blur_image(image)
    tform1 = pre1 + affine1
    distorted_image1 = transform.warp(image, tform1.params, mode='reflect')

    return distorted_image1.astype(np.float32)


# In[ ]:


def assemble_masks(path, id):
    mask = None
    transformed_mask = None
    for i, mask_file in enumerate(glob.glob(path + id + '_*')):
        mask_ = Image.open(mask_file)
        mask_ = np.asarray(mask_)
        if i == 0:
            mask = mask_
            transformed_mask = distort_affine_skimage(mask)
            path2 = 'new_aug/Annotation/'
            new_mask1_path = '%s/%s_rot1_%s.png' % (path2, id, i+1)
            io.imsave(new_mask1_path, transformed_mask)
            continue
        mask = mask | mask_
        transformed_mask = distort_affine_skimage(mask)
        path2 = 'new_aug/Annotation/'
        new_mask1_path = '%s/%s_rot1_%s.png' % (path2, id, i+1)
        io.imsave(new_mask1_path, transformed_mask)


# In[ ]:


def assemble_gland_masks(path, id):
    mask = None
    transformed_mask = None
    for i, mask_file in enumerate(glob.glob(path + id + '_*')):
        mask_ = Image.open(mask_file)
        mask_ = np.asarray(mask_)
        if i == 0:
            mask = mask_
            transformed_mask = distort_affine_skimage(mask)
            path2 = 'Annotation/'
            new_mask1_path = '%s/%s_rot1_%s.png' % (path2, id, i+1)
            io.imsave(new_mask1_path, transformed_mask)
            continue
        mask = mask | mask_
        transformed_mask = distort_affine_skimage(mask)
        path2 = 'Annotation_gland/'
        new_mask1_path = '%s/%s_rot1_%s.png' % (path2, id, i+1)
        io.imsave(new_mask1_path, transformed_mask)


# In[ ]:


image_path = 'color_normalized_imgs/'
# num_files_desired = len(os.listdir(image_path))

mask1_path = 'Annotation/'
mask2_path = 'Annotation_gland/'

images_not_transformed = []

# images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

image_ids = next(os.walk(image_path))[2]

num_generated_files = 0
for n, id_ in tqdm(enumerate(image_ids)):
    if id_.endswith(".png"): 
        id = os.path.splitext(id_)[0]
        image_to_transform = sk.io.imread(os.path.join(image_path,id_),plugin='matplotlib')
        transformed_image = None
        transformed_image = random_zoom(image_to_transform)
        assemble_masks(mask1_path, id)
        assemble_gland_masks(mask2_path, id)
        images_not_transformed.append(image_to_transform)
        path1 = 'new_aug/Images'
        new_image_path = '%s/%s_ab.png' % (path1, id)
        io.imsave(new_image_path, transformed_image)


# In[ ]:




