#Works fine 
# identifies modes aka clusters in the data by iteratively shifting data points towards the densest regions.
#The mean shift vector is computed by taking 
# a weighted average of the nearby data points, 
# where the weights are determined by a kernel function defined by the bandwidth.

import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

# Read the image
image_path = 'Segmentations/00a1c5b94cac4d3293b63a0c602cb5f1.png'  # Replace with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image into a 2D array of pixels
pixels = image_rgb.reshape((-1, 3))

# Estimate bandwidth for Mean Shift
bandwidth = estimate_bandwidth(pixels, quantile=0.2, n_samples=500)

# Perform Mean Shift clustering
meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
meanshift.fit(pixels)

# Get cluster labels and cluster centers
labels = meanshift.labels_
centers = meanshift.cluster_centers_

# Replace pixel values with cluster centers
segmented_pixels = centers[labels].reshape(image_rgb.shape)

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(segmented_pixels.astype(np.uint8))
plt.title('Segmented Image')

plt.tight_layout()
plt.show()
