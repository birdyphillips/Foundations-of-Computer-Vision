"""
Process a scanned sticky note of cursive handwriting (hand_written_image.jpeg):
binarize it with adaptive thresholding (handles the uneven shadow lighting in
the photo), then compare the four core morphological operations (dilation,
erosion, opening, and closing) on the resulting text mask.
"""

import os  # standard library module for file/path operations
import cv2  # OpenCV library for computer vision tasks
from matplotlib import pyplot as plt  # used to build the labeled comparison grid

image_path = os.path.join(os.path.dirname(__file__), "images", "hand_written_image.jpeg")
gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # morphology works on single-channel images

if gray is None:  # cv2.imread returns None if the file wasn't found or couldn't be read
    raise FileNotFoundError(f"Could not load image at: {image_path}")  # stop execution with a clear error

# Adaptive thresholding (vs. a single global Otsu cutoff) copes with the shadow
# falling across the sticky note, thresholding each local neighborhood on its own.
binary = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
    blockSize=35, C=15,
)  # THRESH_BINARY_INV makes the dark ink strokes the white foreground

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # small kernel keeps thin cursive strokes intact

dilation = cv2.dilate(binary, kernel, iterations=1)  # thickens strokes, reconnects faint/broken pen lines
erosion = cv2.erode(binary, kernel, iterations=1)  # thins strokes, strips stray speckle noise from the paper
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # erosion then dilation: clears small specks without thinning strokes
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # dilation then erosion: fills tiny gaps inside strokes/letters

output_dir = os.path.join(os.path.dirname(__file__), "output_images")


def save_window(titles, results, window_title, output_name, suptitle):
    """Plot results side by side in their own figure/window and save it to output_images/."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5), num=window_title, constrained_layout=True)
    for ax, title, result in zip(axes, titles, results):
        ax.imshow(result, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=12)
    fig.suptitle(suptitle, fontsize=16)

    output_path = os.path.join(output_dir, output_name)
    fig.savefig(output_path, dpi=150)
    print(f"Saved {window_title} to: {output_path}")


def save_grid(titles, results, rows, cols, window_title, output_name, suptitle):
    """Plot results in a rows x cols grid in their own figure/window and save it to output_images/."""
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), num=window_title, constrained_layout=True)
    for ax, title, result in zip(axes.ravel(), titles, results):
        ax.imshow(result, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=12)
    fig.suptitle(suptitle, fontsize=16)

    output_path = os.path.join(output_dir, output_name)
    fig.savefig(output_path, dpi=150)
    print(f"Saved {window_title} to: {output_path}")


# Window 1: adaptive threshold binarization
save_window(
    ["Original", "Adaptive Threshold"],
    [gray, binary],
    window_title="Handwritten Text: Adaptive Threshold",
    output_name="handwritten_text_adaptive_threshold.jpg",
    suptitle="Cursive Handwriting: Adaptive Threshold Binarization",
)

# Window 2: dilation vs. erosion
save_window(
    ["Original", "Dilation", "Erosion"],
    [gray, dilation, erosion],
    window_title="Handwritten Text: Dilation vs Erosion",
    output_name="handwritten_text_dilation_erosion.jpg",
    suptitle="Cursive Handwriting: Dilation vs Erosion (3x3 Ellipse Kernel)",
)

# Window 3: opening vs. closing
save_window(
    ["Original", "Opening", "Closing"],
    [gray, opening, closing],
    window_title="Handwritten Text: Opening vs Closing",
    output_name="handwritten_text_opening_closing.jpg",
    suptitle="Cursive Handwriting: Opening vs Closing (3x3 Ellipse Kernel)",
)

# Window 4: all results combined
save_grid(
    ["Original", "Adaptive Threshold", "Dilation", "Erosion", "Opening", "Closing"],
    [gray, binary, dilation, erosion, opening, closing],
    rows=2, cols=3,
    window_title="Handwritten Text: All Operations",
    output_name="handwritten_text_morphology_all.jpg",
    suptitle="Cursive Handwriting: All Morphological Operations (3x3 Ellipse Kernel)",
)

plt.show()
