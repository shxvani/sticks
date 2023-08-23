# Not good, but splits everything into different segments
#  separate objects or regions based on watershed lines
# effective for segmenting objects with well-defined boundaries, such as cells, nuclei
# Watershed lines the ridges that separate different catchment basins


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image_path = 'Segmentations/8a0a32cd8e74421bb4e7076cde64ee60.png'  
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to obtain a binary image
_, thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Perform morphological operations to clean up the image
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Find the unknown region using distance transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Subtract sure_fg from sure_bg to get the unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Label markers for watershed
_, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is 1
markers = markers + 1

# Mark the unknown region as 0
markers[unknown == 255] = 0

# Apply the watershed algorithm
markers = cv2.watershed(image, markers)
image_rgb[markers == -1] = [255, 0, 0]

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(markers, cmap='jet')
plt.title('Segmented Image')

plt.tight_layout()
plt.show()
