"""
Compare mean, median, and Gaussian filtering on noisy_pic.jpg (salt-and-pepper noise)
across 3x3, 5x5, and 7x7 kernels, including two Gaussian sigma values per kernel.
Shows each kernel size in its own 1x4 window, plus one combined 3x4 grid
(rows = kernel size, columns = filter type) with row and column labels.
"""

import os  # standard library module for file/path operations
import cv2  # OpenCV library for computer vision tasks
import matplotlib.pyplot as plt  # used to build the labeled 3x4 comparison grid

image_path = os.path.join(os.path.dirname(__file__), "images", "noisy_pic.jpg")
image = cv2.imread(image_path)  # read the image from disk into a NumPy array (BGR order)

if image is None:  # cv2.imread returns None if the file wasn't found or couldn't be read
    raise FileNotFoundError(f"Could not load image at: {image_path}")  # stop execution with a clear error

kernel_sizes = [3, 5, 7]

# Two fixed sigma values reused across every kernel size so only the kernel changes per row.
# sigma=1.0 is close to OpenCV's own auto-computed sigma for a 5x5 kernel
# (0.3*((ksize-1)*0.5-1)+0.8 = 1.1), giving mild smoothing that doesn't just reduce to a
# box filter. sigma=2.0 doubles that spread to show a clearly stronger blur for comparison.
# Since sigma is much smaller than the 7x7 kernel's footprint in both cases, OpenCV never
# has to clip/renormalize the Gaussian weights, so the two sigmas stay meaningfully
# different at every kernel size instead of collapsing to the same result.
sigmas = [1.0, 2.0]

column_titles = ["Mean Filter", "Median Filter", f"Gaussian (sigma={sigmas[0]})", f"Gaussian (sigma={sigmas[1]})"]
row_titles = [f"{k}x{k} Kernel" for k in kernel_sizes]

output_dir = os.path.join(os.path.dirname(__file__), "output_images")

# Filter results per kernel size, computed once and reused for both the per-kernel
# windows and the combined 3x4 grid below.
results_by_kernel = {}
for k in kernel_sizes:
    # Mean filtering: replaces each pixel with the plain average of its neighborhood
    mean_filtered = cv2.blur(image, (k, k))

    # Median filtering: replaces each pixel with the median of its neighborhood (best for impulse noise)
    median_filtered = cv2.medianBlur(image, k)

    # Gaussian filtering: averages neighboring pixels, weighted by distance from the center
    gaussian_low = cv2.GaussianBlur(image, (k, k), sigmaX=sigmas[0])
    gaussian_high = cv2.GaussianBlur(image, (k, k), sigmaX=sigmas[1])

    results_by_kernel[k] = [mean_filtered, median_filtered, gaussian_low, gaussian_high]

for k in kernel_sizes:
    results = results_by_kernel[k]

    # One figure/window per kernel size, holding its own 1x4 row of filter results
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), num=f"{k}x{k} Kernel")

    for col, result in enumerate(results):
        ax = axes[col]
        ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))  # matplotlib expects RGB, OpenCV stores BGR
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(column_titles[col], fontsize=12)

    fig.suptitle(f"Noisy Pic: {k}x{k} Kernel", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    output_path = os.path.join(output_dir, f"noisy_pic_filter_comparison_{k}x{k}.jpg")
    fig.savefig(output_path, dpi=150)
    print(f"Saved {k}x{k} filter comparison row to: {output_path}")

# Combined 3x4 grid: rows = kernel size, columns = filter type, for easy side-by-side comparison
combined_fig, combined_axes = plt.subplots(len(kernel_sizes), 4, figsize=(16, 12), num="3x4 Grid: All Kernels")

for row, k in enumerate(kernel_sizes):
    for col, result in enumerate(results_by_kernel[k]):
        ax = combined_axes[row, col]
        ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(column_titles[col], fontsize=12)
        if col == 0:
            ax.set_ylabel(row_titles[row], fontsize=12)

combined_fig.suptitle("Noisy Pic: Mean vs. Median vs. Gaussian Filtering", fontsize=16)
combined_fig.tight_layout(rect=(0, 0, 1, 0.96))

combined_output_path = os.path.join(output_dir, "noisy_pic_filter_comparison.jpg")
combined_fig.savefig(combined_output_path, dpi=150)
print(f"Saved 3x4 filter comparison grid to: {combined_output_path}")

plt.show()
