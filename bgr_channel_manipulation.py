"""
Import a kitten image into OpenCV and print its pixel matrices to the console.
Extract its Blue, Green, and Red channels separately, turning the single n x n x 3 image into 3 individual n x n matrices.
Merge those 3 channel matrices back together into a single colored n x n x 3 image.
Swap the red channel with the green channel (GRB order) and merge again to see how the color balance shifts.
"""

import os  # standard library module for file/path operations
import argparse  # parse the --full-matrix command-line flag
import cv2  # OpenCV library for computer vision tasks
import numpy as np  # numerical library used to inspect the pixel array
import matplotlib.pyplot as plt  # used to chart each matrix as a labeled, colorbarred image

parser = argparse.ArgumentParser(description="Extract, merge, and swap the BGR channels of an image.")
parser.add_argument(
    "--full-matrix",
    action="store_true",
    help="Print the full raw pixel matrices to the command line (very large output) instead of just the color pixels.",
)
args = parser.parse_args()

np.set_printoptions(threshold=np.inf, linewidth=200)  # print full arrays instead of truncating them

# 1. Import the image
image_path = os.path.join(os.path.dirname(__file__), "images", "kitten_image.jpg")  # full path to the image file
image = cv2.imread(image_path)  # read the image from disk into a NumPy array (BGR order)

if image is None:  # cv2.imread returns None if the file wasn't found or couldn't be read
    raise FileNotFoundError(f"Could not load image at: {image_path}")  # stop execution with a clear error

height, width, channels = image.shape  # dimensions of the pixel matrix


def print_color_pixels(pixel_matrix, label):
    """Print only the pixels that aren't grayscale (where B, G, R aren't all equal), with their location."""
    blue, green, red = pixel_matrix[:, :, 0], pixel_matrix[:, :, 1], pixel_matrix[:, :, 2]
    is_color = ~((blue == green) & (green == red))  # grayscale/black/white pixels have B == G == R
    rows, cols = np.where(is_color)
    print(f"\n{label} - {len(rows)} color pixel(s) out of {pixel_matrix.shape[0] * pixel_matrix.shape[1]} total:")
    for row, col in zip(rows, cols):
        print(f"  ({row}, {col}): BGR={tuple(int(v) for v in pixel_matrix[row, col])}")


# 2. Print summary info about the matrix
print(f"Image shape (height, width, channels): {image.shape}")
print(f"Data type of pixel values: {image.dtype}")
print(f"Total number of pixels: {height * width}")
print(f"Minimum pixel value: {image.min()}, Maximum pixel value: {image.max()}")
print(
    f"\nSummary:\n"
    f"  - Shape: {height} x {width} x {channels}\n"
    f"  - Data type: unsigned 8-bit integers ({image.dtype})\n"
    f"  - Total pixels: {height * width}\n"
    f"  - Intensity range: 0-255 per channel"
)

# 3. Print the color-related pixels and full BGR pixel matrix only when --full-matrix is passed
if args.full_matrix:
    print("\nFull image pixel matrix (BGR values):")
    print(image)
    print_color_pixels(image, "Original image")

# OpenCV stores each pixel as [Blue, Green, Red] (not RGB), so call out the mapping with a concrete example
example_pixel = image[0, 0]
print(
    f"\nExample pixel at (row=0, col=0): {example_pixel} "
    f"-> Blue={example_pixel[0]}, Green={example_pixel[1]}, Red={example_pixel[2]}"
)

# 4. Extract each color channel separately -> 3 individual 2D (height x width) matrices
blue_channel, green_channel, red_channel = cv2.split(image)  # each channel is now n x n (no 3rd dimension)
print(f"\nBlue channel shape: {blue_channel.shape}")
print(f"Green channel shape: {green_channel.shape}")
print(f"Red channel shape: {red_channel.shape}")

if args.full_matrix:
    print("\nBlue channel matrix:")
    print(blue_channel)
    print("\nGreen channel matrix:")
    print(green_channel)
    print("\nRed channel matrix:")
    print(red_channel)

print(f"\nExample Blue channel value at (row=0, col=0): Blue={blue_channel[0, 0]}")
print(f"Example Green channel value at (row=0, col=0): Green={green_channel[0, 0]}")
print(f"Example Red channel value at (row=0, col=0): Red={red_channel[0, 0]}")

# 5. Print a small, easy-to-read sample block from the top-left corner
sample_region = image[0:5, 0:5]
print("\nTop-left 5x5 pixel block (BGR values):")
print(sample_region)

# 6. Merge the 3 separate 2D channel matrices back into a single colored 3D image
merged_image = cv2.merge([blue_channel, green_channel, red_channel])  # rebuilds the n x n x 3 BGR image
print(f"\nMerged image shape (height, width, channels): {merged_image.shape}")
print(f"Merged image matches original image: {np.array_equal(merged_image, image)}")

merged_pixel = merged_image[0, 0]
print(
    f"\nExample merged pixel at (row=0, col=0): {tuple(int(v) for v in merged_pixel)} "
    f"-> Blue={merged_pixel[0]}, Green={merged_pixel[1]}, Red={merged_pixel[2]}"
)

if args.full_matrix:
    print_color_pixels(merged_image, "Merged image")
    print("\nMerged image matrix (BGR values):")
    print(merged_image)

# 7. Swap the red and green channels and merge again (channel order becomes Green, Red, Blue)
swapped_image = cv2.merge([blue_channel, red_channel, green_channel])  # (B, R, G) order
print(f"\nRed/Green swapped image shape (height, width, channels): {swapped_image.shape}")

swapped_pixel = swapped_image[0, 0]
print(
    f"\nExample swapped pixel at (row=0, col=0): {tuple(int(v) for v in swapped_pixel)} "
    f"-> Blue={swapped_pixel[0]}, Red={swapped_pixel[1]}, Green={swapped_pixel[2]}"
)

if args.full_matrix:
    print_color_pixels(swapped_image, "Red/Green swapped image (slot order is Blue, Red, Green)")
    print("\nRed/Green swapped image matrix (slot order is Blue, Red, Green):")
    print(swapped_image)

# 8. Chart each matrix from this exercise: the 3 extracted channels, the merged image, and the swapped image
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Channel Extraction, Merge, and Swap - Matrix Charts")
print(
    "\nThe figure above uses a 2 x 3 Matplotlib layout: the first row shows the three "
    "color-coded channel matrices (Blue, Green, Red), followed by the merged and swapped "
    "images side by side on the second row."
)

channel_charts = [
    ("Blue Channel", blue_channel, "Blues"),
    ("Green Channel", green_channel, "Greens"),
    ("Red Channel", red_channel, "Reds"),
]
for ax, (title, matrix, cmap) in zip(axes[0], channel_charts):
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=255)
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

merged_rgb = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB)  # matplotlib expects RGB order, OpenCV stores BGR
axes[1, 0].imshow(merged_rgb)
axes[1, 0].set_title("Merged Image (Blue + Green + Red)")
axes[1, 0].set_xlabel("Column")
axes[1, 0].set_ylabel("Row")

swapped_rgb = cv2.cvtColor(swapped_image, cv2.COLOR_BGR2RGB)  # still shows the shifted color balance
axes[1, 1].imshow(swapped_rgb)
axes[1, 1].set_title("Red/Green Swapped Image")
axes[1, 1].set_xlabel("Column")
axes[1, 1].set_ylabel("Row")

axes[1, 2].axis("off")  # unused subplot slot

fig.tight_layout()

# 8b. Separate charts: channels only, merged only, and swapped only
fig_channels, channel_axes = plt.subplots(1, 3, figsize=(15, 5))
fig_channels.suptitle("Extracted Channels")
for ax, (title, matrix, cmap) in zip(channel_axes, channel_charts):
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=255)
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig_channels.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig_channels.tight_layout()

fig_merged, merged_ax = plt.subplots(figsize=(6, 5))
fig_merged.suptitle("Merged Image (Blue + Green + Red)")
merged_ax.imshow(merged_rgb)
merged_ax.set_xlabel("Column")
merged_ax.set_ylabel("Row")
fig_merged.tight_layout()

fig_merged_vs_swapped, merged_vs_swapped_axes = plt.subplots(1, 2, figsize=(10, 5))
fig_merged_vs_swapped.suptitle("Merged vs. Red/Green Swapped")
merged_vs_swapped_axes[0].imshow(merged_rgb)
merged_vs_swapped_axes[0].set_title("Merged Image")
merged_vs_swapped_axes[0].set_xlabel("Column")
merged_vs_swapped_axes[0].set_ylabel("Row")
merged_vs_swapped_axes[1].imshow(swapped_rgb)
merged_vs_swapped_axes[1].set_title("Red/Green Swapped Image")
merged_vs_swapped_axes[1].set_xlabel("Column")
merged_vs_swapped_axes[1].set_ylabel("Row")
fig_merged_vs_swapped.tight_layout()

plt.show()  # blocks until every chart window is closed, then the OpenCV windows below open

# 9. Display each extracted channel as its own 2D grayscale image, plus the merged results
cv2.namedWindow("Blue Channel", cv2.WINDOW_NORMAL)
cv2.imshow("Blue Channel", blue_channel)
cv2.moveWindow("Blue Channel", 100, 100)

cv2.namedWindow("Green Channel", cv2.WINDOW_NORMAL)
cv2.imshow("Green Channel", green_channel)
cv2.moveWindow("Green Channel", 100 + width + 40, 100)

cv2.namedWindow("Red Channel", cv2.WINDOW_NORMAL)
cv2.imshow("Red Channel", red_channel)
cv2.moveWindow("Red Channel", 100 + 2 * (width + 40), 100)

cv2.namedWindow("Merged Color Image", cv2.WINDOW_NORMAL)
cv2.imshow("Merged Color Image", merged_image)
cv2.moveWindow("Merged Color Image", 100 + 3 * (width + 40), 100)

cv2.namedWindow("Red/Green Swapped Image", cv2.WINDOW_NORMAL)
cv2.imshow("Red/Green Swapped Image", swapped_image)
cv2.moveWindow("Red/Green Swapped Image", 100 + 4 * (width + 40), 100)

window_names = [
    "Blue Channel",
    "Green Channel",
    "Red Channel",
    "Merged Color Image",
    "Red/Green Swapped Image",
]
try:
    # waitKey must run at least once before checking window visibility, since imshow needs a
    # pumped event loop to actually paint/register the windows as visible.
    while True:
        if cv2.waitKey(50) != -1:  # a key was pressed (short timeout also lets Ctrl+C interrupt)
            break
        if not any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1 for name in window_names):
            break  # every window has been closed
except KeyboardInterrupt:
    pass
cv2.destroyAllWindows()
