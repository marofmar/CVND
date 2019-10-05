import numpy as np 
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import cv2

image = mpimg.imread('tigre.jpg') 
print('Image dimensions: ', image.shape)

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # change from color to greyscale 
plt.imshow(gray_image, cmap = 'gray') 

x = 400 
y = 300 
print(gray_image[y, x])

max_val = np.amax(gray_image) 
min_val = np.amin(gray_image) 
print('Max: ', max_val) 
print('Min: ', min_val)

'''
RGC channel
'''
rgb_image = mpimg.imread('tigre.jpg')
plt.imshow(image) 

r = rgb_image[:, :, 0]
g = rgb_image[:, :, 1]
b = rgb_image[:, :, 2] 

#visualize individual color channel 
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20,10)) 
ax1.set_title('R channel') 
ax1.imshow(r, cmap = 'gray') 
ax2.set_title('G channel') 
ax2.imshow(g, cmap = 'gray') 
ax3.set_title('B channel') 
ax3.imshow(b, cmap = 'gray') 