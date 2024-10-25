import cv2
import numpy as np

# Load the CAPTCHA image
img = cv2.imread('patrick-files/eoin-test/image.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply median blur to reduce noise while preserving edges
gray_blurred = cv2.medianBlur(gray, 3)

# Binarize the image using Otsu's thresholding
_, thresh = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Remove small noise by filtering out small connected components
# Find all connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

# Create an output image to store the filtered result
filtered = np.zeros(thresh.shape, dtype=np.uint8)

# Set the minimum area for connected components to be considered as characters
min_area = 50  # You may need to adjust this value

# Filter out small components that are likely noise
for i in range(1, num_labels):  # Skip the background label 0
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= min_area:
        filtered[labels == i] = 255
print('exectuted')
# Save the preprocessed image
cv2.imwrite('patrick-files/preprocessing_testing/captcha_preprocessed.png', filtered)
