"""
Segment the gold-and-black desk globe across three scenes (close-up, indoor,
outdoor), following the OpenCV thresholding tutorial's global vs. adaptive
mean vs. adaptive Gaussian recipe, to show why adaptive thresholding copes
better with the uneven/mixed lighting in each photo than one fixed cutoff.

Reference:
OpenCV. (n.d.). Image Thresholding. OpenCV.
https://docs.opencv.org/4.13.0/d7/d4d/tutorial_py_thresholding.html
"""

import os  # standard library module for file/path operations
import cv2  # OpenCV library for computer vision tasks
from matplotlib import pyplot as plt  # used to build the labeled comparison grid

images_dir = os.path.join(os.path.dirname(__file__), "images")
output_dir = os.path.join(os.path.dirname(__file__), "output_images")

# (filename, display label, global threshold v) tuned per scene:
#  - close-up: busy indoor background behind the globe, globe fills most of the frame
#  - indoor: even room lighting, globe is small against a dark wood table
#  - outdoor: strong backlight/sun makes the globe a near-silhouette against a bright field
scenes = [  # list of (filename, label, global_v) tuples, one per scene
    ("globe_close_up_scene.jpg", "Close-Up", 127),
    ("globe_indoor_scene.jpg", "Indoor", 127),
    ("globe_outdoor_scene.jpg", "Outdoor", 127),
]

print(f"{'Scene':<10}{'Global':>10}{'Adaptive Mean':>16}{'Adaptive Gaussian':>20}")
print("-" * 56)

saved_messages = []  # collected during the loop, printed together after the table
patch_messages = []  # raw grayscale pixel samples, collected during the loop, printed together after the table
for filename, label, global_v in scenes:  # repeat the same thresholding recipe for each scene photo
    image_path = os.path.join(images_dir, filename)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:  # cv2.imread returns None if the file wasn't found or couldn't be read
        raise FileNotFoundError(f"Could not load image at: {image_path}")  # stop execution with a clear error

    img = cv2.medianBlur(img, 5)

    # blockSize scales with image size so the local neighborhood stays proportional across the 3 resolutions
    # (the tutorial's fixed 11 is sized for the small sudoku.png, too tight a window for these full-size photos).
    block_size = max(11, (min(img.shape) // 20) | 1)  # "| 1" forces the value odd, as adaptiveThreshold requires

    _, th1 = cv2.threshold(img, global_v, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 2)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2)

    # % of pixels pushed to white (255) in each binary result, as a quick numeric read on how much each method segmented out
    global_pct = 100 * (th1 == 255).mean()
    mean_pct = 100 * (th2 == 255).mean()
    gaussian_pct = 100 * (th3 == 255).mean()
    print(f"{label:<10}{global_pct:>9.1f}%{mean_pct:>15.1f}%{gaussian_pct:>19.1f}%")

    # raw grayscale values (0-255) in a 5x5 patch at the image center, i.e. what adaptiveThreshold actually compares
    cy, cx = img.shape[0] // 2, img.shape[1] // 2
    patch = img[cy - 2:cy + 3, cx - 2:cx + 3]
    patch_messages.append(f"{label} raw pixel values (5x5 patch at center, row={cy}, col={cx}):\n{patch}")

    titles = [
        "Original Image",
        f"Global Thresholding (v = {global_v})",
        "Adaptive Mean Thresholding",
        "Adaptive Gaussian Thresholding",
    ]
    images = [img, th1, th2, th3]

    fig = plt.figure(figsize=(10, 10), num=f"Globe {label} Scene")
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], "gray")
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    fig.suptitle(f"Globe {label} Scene: Global vs. Adaptive Thresholding", fontsize=14)

    output_path = os.path.join(output_dir, f"adaptive_threshold_{os.path.splitext(filename)[0]}.jpg")
    fig.savefig(output_path, dpi=150)
    saved_messages.append(f"Saved {label} comparison to: {output_path}")

print()
print("\n\n".join(patch_messages))

print()
print("\n".join(saved_messages))

plt.show()
