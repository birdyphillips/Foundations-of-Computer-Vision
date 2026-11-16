"""
Demonstrate morphological operations (erosion, dilation, gradient, open, close,
top hat, black hat, hit-or-miss) on morph_input.jpg, plus print the three
structuring element shapes OpenCV can generate (rect, ellipse, cross).
"""

import os  # standard library module for file/path operations
import numpy as np  # used to build kernels
import cv2  # OpenCV library for computer vision tasks
from matplotlib import pyplot as plt  # used to build the labeled subplot grids

output_dir = os.path.join(os.path.dirname(__file__), "output_images")
image_filenames = ["morph_input.jpg"]


def save_grid(results, titles, rows, cols, window_title, output_name):
    """Plot results in a rows x cols grid of titled, axis-free images and save it to output_images/."""
    fig = plt.figure(window_title, figsize=(cols * 3, rows * 3.5))
    for i, (result, title) in enumerate(zip(results, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(result, cmap="gray")
        plt.title(title)
        plt.axis("off")

    output_path = os.path.join(output_dir, output_name)
    fig.savefig(output_path, dpi=150)
    print(f"Saved {window_title} grid to: {output_path}")


# --- Structuring elements: rect, ellipse, and cross shapes, each 5x5 ---
print("Rectangular structuring element (5x5):")
print(cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
print("\nElliptical structuring element (5x5):")
print(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
print("\nCross structuring element (5x5):")
print(cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)))

for image_filename in image_filenames:
    image_path = os.path.join(os.path.dirname(__file__), "images", image_filename)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # morphology works on single-channel images

    if gray is None:  # cv2.imread returns None if the file wasn't found or couldn't be read
        raise FileNotFoundError(f"Could not load image at: {image_path}")  # stop execution with a clear error

    # Otsu thresholding picks a cutoff automatically so morphology has a true binary image to work with
    _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    base_name = os.path.splitext(image_filename)[0]

    # --- Erosion, dilation, and gradient with a plain 3x3 square kernel ---
    square_kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img, square_kernel, iterations=1)  # shrinks white regions, trims their edges
    dilation = cv2.dilate(img, square_kernel, iterations=1)  # grows white regions, thickens their edges
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, square_kernel)  # dilation minus erosion = outline

    # --- Open, close, top hat, black hat, and hit-or-miss with a 3x3 cross kernel ---
    cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, cross_kernel)  # erosion then dilation: removes small white specks
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cross_kernel)  # dilation then erosion: fills small black gaps
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cross_kernel)  # original minus opening: isolates small bright details
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, cross_kernel)  # closing minus original: isolates small dark details
    hitmiss = cv2.morphologyEx(img, cv2.MORPH_HITMISS, cross_kernel)  # matches a specific foreground/background pattern

    save_grid(
        [img, erosion, dilation, gradient, opening, closing, tophat, blackhat, hitmiss],
        ["Original", "Erosion", "Dilation", "Gradient", "Open", "Close", "Top hat", "Black hat", "Hit Miss"],
        rows=3, cols=3,
        window_title=f"Morphological Operations - {base_name}",
        output_name=f"morphological_ops_{base_name}.jpg",
    )

plt.show()
