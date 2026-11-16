"""
Copy of bgr_channel_manipulation.py with extra table-image output for taking clean screenshots.
Not for submission -- just a personal helper to generate screenshot-friendly PNGs of the sample matrices.
"""

import os  # standard library module for file/path operations
import cv2  # OpenCV library for computer vision tasks
import numpy as np  # numerical library used to inspect the pixel array
import matplotlib.pyplot as plt  # used to render small matrices as clean, screenshot-friendly table images

np.set_printoptions(threshold=np.inf, linewidth=200)  # print full arrays instead of truncating them

output_dir = os.path.join(os.path.dirname(__file__), "matrix_screenshots")  # folder for the table images
os.makedirs(output_dir, exist_ok=True)


def save_matrix_table(title, matrix, filename, row_offset=0, col_offset=0):
    """Render a small 2D matrix as a labeled table image and save it as a PNG."""
    row_labels = [f"Row {row_offset + r}" for r in range(matrix.shape[0])]
    col_labels = [f"Col {col_offset + c}" for c in range(matrix.shape[1])]
    cell_text = [[str(value) for value in row] for row in matrix]

    fig, ax = plt.subplots(figsize=(matrix.shape[1] * 0.9 + 1, matrix.shape[0] * 0.5 + 1))
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=12)
    table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc="center", cellLoc="center")
    table.scale(1, 1.6)
    fig.tight_layout()

    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved table image: {output_path}")


def save_combined_matrix_tables(rows, filename, suptitle=None, figsize=None):
    """Render several small 2D matrices as labeled tables arranged in a grid (list of rows) and save as one PNG."""
    n_rows = len(rows)
    n_cols = max(len(row) for row in rows)
    if figsize is None:
        col_width = max(matrix.shape[1] for row in rows for _, matrix in row) * 1.2 + 1
        row_height = max(matrix.shape[0] for row in rows for _, matrix in row) * 0.6 + 2
        figsize = (col_width * n_cols, row_height * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)
    if suptitle:
        fig.suptitle(suptitle)

    for r, row in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(row):
                ax.axis("off")  # unused grid slot (e.g. a shorter bottom row)
                continue
            title, matrix = row[c]
            row_labels = [f"Row {i}" for i in range(matrix.shape[0])]
            col_labels = [f"Col {i}" for i in range(matrix.shape[1])]
            cell_text = [[str(value) for value in matrix_row] for matrix_row in matrix]
            ax.axis("off")
            ax.set_title(title, fontsize=13, pad=2)
            table = ax.table(
                cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc="upper center", cellLoc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(13)
            table.scale(1, 3.2)

    fig.tight_layout(pad=1.0, h_pad=0.3, w_pad=0.5)
    fig.subplots_adjust(hspace=0.15)
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined table image: {output_path}")

# 1. Import the image
image_path = os.path.join(os.path.dirname(__file__), "images", "kitten_image.jpg")  # full path to the image file
image = cv2.imread(image_path)  # read the image from disk into a NumPy array (BGR order)

if image is None:  # cv2.imread returns None if the file wasn't found or couldn't be read
    raise FileNotFoundError(f"Could not load image at: {image_path}")  # stop execution with a clear error

height, width, channels = image.shape  # dimensions of the pixel matrix

# 2. Print summary info about the matrix
print(f"Image shape (height, width, channels): {image.shape}")
print(f"Data type of pixel values: {image.dtype}")
print(f"Total number of pixels: {height * width}")
print(f"Minimum pixel value: {image.min()}, Maximum pixel value: {image.max()}")

# 3. Print summary info only -- the full matrix print was removed since it's extremely slow for large images

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

# (full channel matrix prints removed -- see the 5x5 screenshot table images instead)

print(f"\nExample Blue channel value at (row=0, col=0): Blue={blue_channel[0, 0]}")
print(f"Example Green channel value at (row=0, col=0): Green={green_channel[0, 0]}")
print(f"Example Red channel value at (row=0, col=0): Red={red_channel[0, 0]}")

# 5. Print a small, easy-to-read sample block from the top-left corner
sample_region = image[0:5, 0:5]
print("\nTop-left 5x5 pixel block (BGR values):")
print(sample_region)

# Save screenshot-friendly table images of the top-left 5x5 block for each channel
save_matrix_table("Blue Channel - Top-left 5x5", blue_channel[0:5, 0:5], "blue_channel_5x5.png")
save_matrix_table("Green Channel - Top-left 5x5", green_channel[0:5, 0:5], "green_channel_5x5.png")
save_matrix_table("Red Channel - Top-left 5x5", red_channel[0:5, 0:5], "red_channel_5x5.png")

# 6. Merge the 3 separate 2D channel matrices back into a single colored 3D image
merged_image = cv2.merge([blue_channel, green_channel, red_channel])  # rebuilds the n x n x 3 BGR image
print(f"\nMerged image shape (height, width, channels): {merged_image.shape}")
print(f"Merged image matches original image: {np.array_equal(merged_image, image)}")

merged_pixel = merged_image[0, 0]
print(
    f"\nExample merged pixel at (row=0, col=0): {tuple(int(v) for v in merged_pixel)} "
    f"-> Blue={merged_pixel[0]}, Green={merged_pixel[1]}, Red={merged_pixel[2]}"
)

save_matrix_table("Merged Image - Top-left 5x5 (BGR)", merged_image[0:5, 0:5], "merged_image_5x5.png")

# 7. Swap the red and green channels and merge again (channel order becomes Green, Red, Blue)
swapped_image = cv2.merge([blue_channel, red_channel, green_channel])  # (B, R, G) order
print(f"\nRed/Green swapped image shape (height, width, channels): {swapped_image.shape}")

swapped_pixel = swapped_image[0, 0]
print(
    f"\nExample swapped pixel at (row=0, col=0): {tuple(int(v) for v in swapped_pixel)} "
    f"-> Blue={swapped_pixel[0]}, Red={swapped_pixel[1]}, Green={swapped_pixel[2]}"
)

save_matrix_table("Red/Green Swapped Image - Top-left 5x5 (B, R, G)", swapped_image[0:5, 0:5], "swapped_image_5x5.png")

# Combine all 5 matrix table screenshots (Blue, Green, Red, Merged, Swapped) into a single PNG
save_combined_matrix_tables(
    [
        [
            ("Blue Channel", blue_channel[0:5, 0:5]),
            ("Green Channel", green_channel[0:5, 0:5]),
            ("Red Channel", red_channel[0:5, 0:5]),
        ],
        [
            ("Merged Image (Blue + Green + Red)", merged_image[0:5, 0:5]),
            ("Red/Green Swapped Image", swapped_image[0:5, 0:5]),
        ],
    ],
    "combined_matrix_tables_5x5.png",
    suptitle="Channel Extraction, Merge, and Swap - Matrix Tables",
)
print(
    "\nFigure 6 (combined_matrix_tables_5x5.png) uses a 2 x 3 Matplotlib layout: the top row shows "
    "the top-left 5x5 block of each color-coded channel matrix (Blue, Green, Red), and the bottom "
    "row shows the same 5x5 block for the merged image and the red/green swapped image."
)

# 8. Display each extracted channel as its own 2D grayscale image, plus the merged results
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
