# Not good 3/10 for effort
# (Simple Linear Iterative Clustering) 
# group pixels into perceptually meaningful and relatively uniform regions, which are called superpixels.
# SLIC is a simple extension of k-means clustering in the pixel space,

import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.io import imread

# Read the image
image_path = 'Segmentations/8a0a32cd8e74421bb4e7076cde64ee60.png'  # Replace with your image path
image = imread(image_path)
image_rgb = image.copy()

# Perform SLIC superpixel segmentation
segments = slic(image_rgb, n_segments=100, compactness=10, sigma=1)
# compact is balance between color and spatial proximity, sigma is smoothness
# Create a mask for each segment
segment_mask = np.zeros_like(image)
for segment_label in np.unique(segments):
    segment_mask[segments == segment_label] = segment_label

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(segment_mask, cmap='tab20')
plt.title('Segmented Image')

plt.tight_layout()
plt.show()
