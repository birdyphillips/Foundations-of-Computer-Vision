"""
Apply histogram equalization to brain_image.jpg to improve contrast in dark/low-contrast regions,
then save and display the original and equalized images side by side for comparison.
"""

import os  # standard library module for file/path operations
import cv2  # OpenCV library for computer vision tasks
import numpy as np  # numerical library used to stack the images side by side

# 1. Import the image in grayscale (histogram equalization works on single-channel intensity values)
image_path = os.path.join(os.path.dirname(__file__), "images", "brain_image.jpg")  # full path to the image file
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # read the image from disk as grayscale

if image is None:  # cv2.imread returns None if the file wasn't found or couldn't be read
    raise FileNotFoundError(f"Could not load image at: {image_path}")  # stop execution with a clear error

# 2. Apply histogram equalization to spread out the pixel intensities across the full 0-255 range
equalized = cv2.equalizeHist(image)

# 3. Stack the original and equalized images side by side for a direct comparison
comparison = np.hstack((image, equalized))

# 4. Save the comparison image
output_path = os.path.join(os.path.dirname(__file__), "output_images", "brain_image_equalized.jpg")
cv2.imwrite(output_path, comparison)
print(f"Saved original vs. equalized comparison to: {output_path}")

# 5. Display the result so it can be visually confirmed
cv2.namedWindow("Original (left) vs Equalized (right)", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.imshow("Original (left) vs Equalized (right)", comparison)
cv2.moveWindow("Original (left) vs Equalized (right)", 100, 100)

while cv2.getWindowProperty("Original (left) vs Equalized (right)", cv2.WND_PROP_VISIBLE) >= 1:  # keep waiting until the window is closed
    if cv2.waitKey(50) != -1:  # exit early if a key is pressed while the window has focus
        break
cv2.destroyAllWindows()
