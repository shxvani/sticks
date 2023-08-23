#Works, 8.5/10
#groups pixels into segments based on their similarity in color and intensity
#Graph undirected weighted with each pixel as a node, low weight is close affinity, Iteration.
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb
from skimage.color import rgb2gray
from skimage.io import imread

# Read the image
image_path = 'Segmentations/8a0a32cd8e74421bb4e7076cde64ee60.png'  # Replace with your image path
image = imread(image_path)
image_rgb = image.copy()

# Convert the image to grayscale
gray_image = rgb2gray(image)

# Perform Felzenszwalb's segmentation
segments = felzenszwalb(image_rgb, scale=100, sigma=0.5, min_size=50)

# Create a mask for each segment
segment_mask = np.zeros_like(gray_image)
for segment_label in np.unique(segments):
    segment_mask[segments == segment_label] = segment_label

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(segment_mask, cmap='jet')
plt.title('Segmented Image')

plt.tight_layout()
plt.show()
