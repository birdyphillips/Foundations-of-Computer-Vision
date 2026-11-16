"""
Compare the basic morphological operations (erosion, dilation, gradient, open,
close, top hat, black hat) on morph_input.jpg using a single 5x5 kernel, shown
as a labeled 2x4 grid.
"""

import os  # standard library module for file/path operations
import numpy as np  # used to build the square kernel
import cv2  # OpenCV library for computer vision tasks
import matplotlib.pyplot as plt  # used to build the labeled comparison grid

image_path = os.path.join(os.path.dirname(__file__), "images", "morph_input.jpg")
gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # morphology works on single-channel images

if gray is None:  # cv2.imread returns None if the file wasn't found or couldn't be read
    raise FileNotFoundError(f"Could not load image at: {image_path}")  # stop execution with a clear error

# Otsu thresholding picks a cutoff automatically so morphology has a true binary image to work with
_, image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((5, 5), np.uint8)

erosion = cv2.erode(image, kernel, iterations=1)  # shrinks white regions, trims their edges
dilation = cv2.dilate(image, kernel, iterations=1)  # grows white regions, thickens their edges
gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)  # dilation minus erosion = outline
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)  # erosion then dilation: removes small white specks
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  # dilation then erosion: fills small black gaps
tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)  # original minus opening: isolates small bright details
blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)  # closing minus original: isolates small dark details

titles = ["Original", "Erosion", "Dilation", "Gradient", "Open", "Close", "Top Hat", "Black Hat"]
results = [image, erosion, dilation, gradient, opening, closing, tophat, blackhat]

fig, axes = plt.subplots(2, 4, figsize=(16, 8), num="Morph Input: Basic Operations")

for i, (title, result) in enumerate(zip(titles, results)):
    ax = axes[i // 4, i % 4]
    ax.imshow(result, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)

fig.suptitle("Morph Input: Basic Morphological Operations (5x5 Kernel)", fontsize=16)
fig.tight_layout(rect=(0, 0, 1, 0.94))

output_dir = os.path.join(os.path.dirname(__file__), "output_images")
output_path = os.path.join(output_dir, "morph_input_operations_comparison.jpg")
fig.savefig(output_path, dpi=150)
print(f"Saved basic operations comparison grid to: {output_path}")

plt.show()
