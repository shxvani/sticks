# Works, for threshold 100 => 8.5/10 
# 200 => 6.5/10
# Identify abrupt changes in intensity or color in an image. =>Sharp discontinuities in intensity values

from PIL import Image, ImageOps, ImageFilter

# Read the image
image_path = '/Users/shiv/projectsem7/Segmentations/1a0a2b3870a64e43a2fd986a075aee8f.png'  # Replace with your image path
image = Image.open(image_path)

# Convert to grayscale
gray_image = ImageOps.grayscale(image)

# Apply thresholding to segment the image
threshold_value = 160
binary_image = gray_image.point(lambda p: p > threshold_value and 255)

# Find contours using a simple edge detection
contours = binary_image.filter(ImageFilter.FIND_EDGES)

# Create a mask for segmented areas
segmented_mask = contours.point(lambda p: p > 128 and 255)

# Apply the mask to the original image to get the segmented regions
segmented_image = Image.new('RGB', image.size)
segmented_image.paste(image, mask=segmented_mask)

# Display the results
segmented_image.show()
