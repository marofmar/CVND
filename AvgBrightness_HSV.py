# -*- coding: utf-8 -*-
"""CVND_L4_Standardize_AvgBrightness.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18n-tjNZz-heTsQsreuFtpPl94hxkK1qf
"""

# Commented out IPython magic to ensure Python compatibility.
import cv2 # computer vision library

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils

# %matplotlib inline

night = mpimg.imread('nvd.JPG')
plt.imshow(night)

day = mpimg.imread('IMG_4574.JPG')
plt.imshow(day)

print(day.shape, night.shape)

"""# Input Standardize"""

def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def standardize_input(image):
    std_img = cv2.resize(image, (1200,900))
    return std_img

plt.imshow(standardize_input(rotateImage(day, 270)))

plt.imshow(standardize_input(rotateImage(night, 270)))

"""# Feature Extraction
## RGB to HSV conversion
### For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
"""

day_img = standardize_input(day)
night_img = standardize_input(night)

hsv = cv2.cvtColor(day_img, cv2.COLOR_RGB2HSV)
print('DAY image') 

#hsv channel
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

# plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (20,10))
ax1.set_title('the original image')
ax1.imshow(day_img) 
ax2.set_title('H channel')
ax2.imshow(h, cmap = 'gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap = 'gray') 
ax4.set_title('V channel')
ax4.imshow(v, cmap = 'gray')

hsv2 = cv2.cvtColor(night_img, cv2.COLOR_RGB2HSV)
print('NIGHT image') 

#hsv channel
h2 = hsv2[:,:,0]
s2 = hsv2[:,:,1]
v2 = hsv2[:,:,2]

# plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (20,10))
ax1.set_title('the original image')
ax1.imshow(night_img) 
ax2.set_title('H channel')
ax2.imshow(h2, cmap = 'gray')
ax3.set_title('S channel')
ax3.imshow(s2, cmap = 'gray') 
ax4.set_title('V channel')
ax4.imshow(v2, cmap = 'gray')

"""- Based on the two distinguishly diff results of V channel, 'value' would be the key to classify 'day' and 'night'

# Average brightness using the V channel
"""

def avg_brightness(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    sum_brightness = np.sum(hsv[:,:,2]) # sum the V values 
    area = rgb_image.shape[0] * rgb_image.shape[1]
    avg = sum_brightness / area
    #plt.imshow(avg)
    return avg

avg_brightness(day_img)

avg_brightness(night_img)
