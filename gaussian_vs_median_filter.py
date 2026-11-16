"""
Compare Gaussian blurring and median filtering on two noisy images - noisy_pic.jpg
(salt-and-pepper noise) and image-after-gaussian-noise-1.jpg (Gaussian noise) -
then save and display the original, Gaussian-blurred, and median-filtered images side by side.
"""

import os  # standard library module for file/path operations
import cv2  # OpenCV library for computer vision tasks
import numpy as np  # numerical library used to stack the images side by side


def label_image(img, text):
    """Return a copy of img with a text label drawn in the top-left corner."""
    labeled = img.copy()
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
    position = (10, 30)
    # Draw a thick black outline first, then white on top, so the label stays readable on any background
    cv2.putText(labeled, text, position, font, font_scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(labeled, text, position, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return labeled


def build_comparison(image_filename, noise_label):
    """Load image_filename, apply Gaussian/median filters, and return the labeled side-by-side comparison."""
    image_path = os.path.join(os.path.dirname(__file__), "images", image_filename)
    image = cv2.imread(image_path)  # read the image from disk into a NumPy array (BGR order)

    if image is None:  # cv2.imread returns None if the file wasn't found or couldn't be read
        raise FileNotFoundError(f"Could not load image at: {image_path}")  # stop execution with a clear error

    # Gaussian blurring: averages neighboring pixels, weighted by distance from the center
    gaussian = cv2.GaussianBlur(image, (5, 5), sigmaX=0)

    # Median filtering: replaces each pixel with the median of its neighborhood
    median = cv2.medianBlur(image, 5)

    return np.hstack((
        label_image(image, f"Original ({noise_label})"),
        label_image(gaussian, "Gaussian Blur"),
        label_image(median, "Median Filter"),
    ))


# Each entry: source image, its noise type (for the on-image label), and the saved comparison filename
sources = [
    ("noisy_pic.jpg", "Salt-and-Pepper Noise", "noisy_pic_filter_comparison.jpg"),
    ("image-after-gaussian-noise-1.jpg", "Gaussian Noise", "gaussian_noise_filter_comparison.jpg"),
    ("uneven_lighting.jpeg", "Uneven Lighting", "uneven_lighting_filter_comparison.jpg"),
]

output_dir = os.path.join(os.path.dirname(__file__), "output_images")
window_names = []

for index, (image_filename, noise_label, output_filename) in enumerate(sources):
    comparison = build_comparison(image_filename, noise_label)

    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, comparison)
    print(f"Saved {noise_label} original vs. Gaussian vs. median comparison to: {output_path}")

    window_name = f"{noise_label}: Original | Gaussian Blur | Median Filter"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    height, width = comparison.shape[:2]
    display_height = 500  # scale the window to a reasonable on-screen size while keeping the full-res file untouched
    display_width = int(width * display_height / height)
    cv2.resizeWindow(window_name, display_width, display_height)
    cv2.imshow(window_name, comparison)
    cv2.moveWindow(window_name, 100, 100 + index * 550)  # stack windows vertically so both are visible
    window_names.append(window_name)

# Keep waiting until every window has been closed
while any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1 for name in window_names):
    if cv2.waitKey(50) != -1:  # exit early if a key is pressed while a window has focus
        break
cv2.destroyAllWindows()
