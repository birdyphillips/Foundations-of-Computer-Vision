"""
Segment the coins in 3_pennies_.jpeg using the classic OpenCV watershed
recipe: Otsu thresholding for an approximate coin mask, a distance transform
to separate touching coins into sure-foreground peaks, and cv2.watershed()
to find the final boundary from the sure-foreground/background markers.
"""

import os  # standard library module for file/path operations
import numpy as np  # used for marker arrays and the structuring element
import cv2  # OpenCV library for computer vision tasks
from matplotlib import pyplot as plt  # used to build the labeled comparison grid

image_path = os.path.join(os.path.dirname(__file__), "images", "3_pennies_.jpeg")
image = cv2.imread(image_path)  # watershed needs the original color image to draw boundaries on

if image is None:  # cv2.imread returns None if the file wasn't found or couldn't be read
    raise FileNotFoundError(f"Could not load image at: {image_path}")  # stop execution with a clear error

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # watershed thresholding/distance transform need a single channel
gray = cv2.GaussianBlur(gray, (7, 7), 0)  # smooth out embossed coin detail noise without blurring the real gaps between coins
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # coins are brighter than the wood, so no inversion is needed

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)  # noise removal

sure_bg = cv2.dilate(opening, kernel, iterations=3)  # sure background area

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # finding sure foreground area
_, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)  # lower cutoff so the weaker, more worn coin still counts as foreground

sure_fg = np.uint8(sure_fg)  # finding unknown region
unknown = cv2.subtract(sure_bg, sure_fg)

_, markers = cv2.connectedComponents(sure_fg)  # marker labelling

markers = markers + 1  # add one to all labels so that sure background is not 0, but 1

markers[unknown == 255] = 0  # now, mark the region of unknown with zero

markers = cv2.watershed(image, markers)

result = image.copy()
result[markers == -1] = [0, 0, 255]  # paint the detected coin boundaries red (BGR)

num_coins = len(np.unique(markers)) - 2  # exclude the boundary (-1) and background (1) labels
print(f"Coins detected: {num_coins}")

output_dir = os.path.join(os.path.dirname(__file__), "output_images")

titles = ["Original", "Otsu Threshold", "Distance Transform", "Markers", "Watershed Boundaries"]
results = [
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    thresh,
    dist_transform,
    markers,
    cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
]
cmaps = [None, "gray", "gray", "jet", None]

fig, axes = plt.subplots(2, 3, figsize=(15, 10), num="Coin Segmentation: Watershed", constrained_layout=True)

for ax, title, res, cmap in zip(axes.ravel(), titles, results, cmaps):
    ax.imshow(res, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)
axes.ravel()[-1].axis("off")  # 6th grid cell is unused (5 images in a 2x3 grid)

fig.suptitle(f"Coin Segmentation via Watershed ({num_coins} coins detected)", fontsize=16)

output_path = os.path.join(output_dir, "watershed_coin_segmentation.jpg")
fig.savefig(output_path, dpi=150)
print(f"Saved watershed segmentation comparison grid to: {output_path}")

plt.show()
