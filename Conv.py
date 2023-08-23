import numpy as np
import cv2

# Read the image
image = cv2.imread('Segmentations/8a0a32cd8e74421bb4e7076cde64ee60.png', cv2.IMREAD_GRAYSCALE)

# Define a Gaussian blur kernel
kernel = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]])

# Perform convolution
convolved_image = cv2.filter2D(image, -1, kernel)

# Display the original and convolved images
cv2.imshow('Original Image', image)
cv2.imshow('Convolved Image', convolved_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
