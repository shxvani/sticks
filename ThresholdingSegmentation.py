# Okay. Thresholding to separate the background from the foreground  based on their pixel intensity values.
from PIL import Image

# Read the image
image_path = 'Segmentations/1a0a2b3870a64e43a2fd986a075aee8f.png' 
image = Image.open(image_path)
gray_image = image.convert('L') # Convert the image to grayscale

# Get pixel data
pixels = gray_image.load()

# Set a threshold value
threshold_value = 100

# Create a new image for segmented regions
segmented_image = Image.new('RGB', image.size)
segmented_pixels = segmented_image.load()

# Segment the image based on threshold
for x in range(image.width):
    for y in range(image.height):
        if pixels[x, y] > threshold_value:
            segmented_pixels[x, y] = pixels[x, y], pixels[x, y], pixels[x, y]
        else:
            segmented_pixels[x, y] = 255, 255, 255

# Display the results
segmented_image.show()
