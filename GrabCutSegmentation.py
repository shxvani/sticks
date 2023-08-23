#Not effective for this image. 6/10
#user specifies rect, energy minimization function, image is graph like, initially assigned to foreg.
#optimise and refine iteration till convergence


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image_path = 'Segmentations/8a0a32cd8e74421bb4e7076cde64ee60.png'  # Replace with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a mask for the foreground and background
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Create a rectangle around the region of interest (foreground)
rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)

# Initialize the algorithm with the rectangle and mask
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# Apply the GrabCut algorithm
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Modify the mask to separate the definite background from probable background
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Multiply the original image with the mask to get the segmented image
segmented_image = image * mask2[:, :, np.newaxis]

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title('Segmented Image')

plt.tight_layout()
plt.show()
