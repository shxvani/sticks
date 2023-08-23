# 10/10
# Pick no (k) of clusters, and k centroids, assign each data point to nearest point
# update centroid by calc average
# Iterate till convergence


import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Read the image
image_path = 'Segmentations/00a1c5b94cac4d3293b63a0c602cb5f1.png'  # Replace with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image into a 2D array of pixels
pixels = image_rgb.reshape((-1, 3))

# Set the number of clusters (segments)
num_clusters = 2
# num_clusters = 20

# Perform K-Means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(pixels)

# Get cluster labels and cluster centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Replace pixel values with cluster centers
segmented_pixels = centers[labels].reshape(image_rgb.shape)

# Display the original and segmented images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(segmented_pixels.astype(np.uint8))
plt.title('Segmented Image')

plt.tight_layout()
plt.show()
