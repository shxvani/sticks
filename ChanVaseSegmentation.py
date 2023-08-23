# #  Not Working
# Done by detecting object boundaries. Partition an image into regions with homogeneous 
# intensities while maintaining smooth boundaries between the regions. 
# Successful use of the Chan-Vese algorithm depends on proper parameter tuning, 
# including the values of alpha, beta, and gamma. 
# Best for those with clear object boundaries and relatively homogeneous regions.
#initial contour, which can be a simple shape (e.g., a circle) or a more complex shape (e.g., a polygon).
# Compute the average intensities inside and outside the contour
# Regularization term penalizes complex contour shapes and encourages smoothness
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, morphology, segmentation
from skimage.io import imread
from skimage.segmentation import active_contour

# Read the image
image_path = 'Segmentations/0a0a61d5a0e043cc8def74e629f1d465.png'  # Replace with your image path
image = imread(image_path)
image_gray = color.rgb2gray(image)

# Create an initial contour
s = np.linspace(0, 2*np.pi, 400)
r = 150 + 20*np.sin(s)
c = 250 + 40*np.cos(s)
init_contour = np.array([r, c]).T

# Perform active contour segmentation
snake = active_contour(image_gray, init_contour, alpha=0.01, beta=1, gamma=0.001)

# Display the results
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(image, cmap=plt.cm.gray)
ax.plot(init_contour[:, 1], init_contour[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, image.shape[1], image.shape[0], 0])

plt.show()
