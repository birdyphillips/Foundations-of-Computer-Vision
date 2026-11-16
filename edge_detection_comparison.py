import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

immg = np.zeros((512, 512, 3), dtype=np.uint8)
immg[:] = (255, 0, 0)  # OpenCV uses BGR order, so (255,0,0) is blue
cv.rectangle(immg, (100, 100), (400, 400), (0, 255, 0), cv.FILLED)  # (top-left x,y), (bottom-right x,y), (B,G,R) color
cv.circle(immg, (256, 256), 50, (0, 0, 255), cv.FILLED)  # (center x,y), radius, (B,G,R) color

noise = np.random.normal(0, 25, immg.shape)  # Gaussian noise, mean 0, std 25
noisy_immg = np.clip(immg.astype(np.float32) + noise, 0, 255).astype(np.uint8)  # add noise and clip back to valid pixel range

gradient = np.linspace(0.3, 1.0, immg.shape[1]).reshape(1, -1, 1)  # brightness factor rising left to right
varying_immg = np.clip(immg.astype(np.float32) * gradient, 0, 255).astype(np.uint8)  # simulate uneven lighting across the image

# clean, title-free copies so edge_detection_evaluation.py can load the same pixels back from disk
images_dir = os.path.join(os.path.dirname(__file__), "images")
cv.imwrite(os.path.join(images_dir, "edge_detection_source_normal.jpg"), immg)
cv.imwrite(os.path.join(images_dir, "edge_detection_source_noisy.jpg"), noisy_immg)
cv.imwrite(os.path.join(images_dir, "edge_detection_source_varying_intensity.jpg"), varying_immg)


def detect_edges(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Sobel/LoG need a single-channel image
    canny_edges = cv.Canny(image, 100, 200)  # apply Canny edge detection with thresholds 100 and 200
    sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)  # gradient in the x direction
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)  # gradient in the y direction
    sobel_edges = cv.convertScaleAbs(cv.magnitude(sobel_x, sobel_y))  # combine x/y gradients into one edge map
    blurred = cv.GaussianBlur(gray, (3, 3), 0)  # smooth noise before taking the Laplacian
    log_edges = cv.convertScaleAbs(cv.Laplacian(blurred, cv.CV_64F, ksize=3))  # LoG: Gaussian blur then Laplacian
    return canny_edges, sobel_edges, log_edges


def save_comparison(image, label, filename):
    canny_edges, sobel_edges, log_edges = detect_edges(image)

    plt.figure(figsize=(20, 5))  # wide canvas so a 1x4 row of 512x512 images is still readable
    plt.subplot(141),plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title(label)
    plt.subplot(142),plt.imshow(canny_edges, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.subplot(143),plt.imshow(sobel_edges, cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.subplot(144),plt.imshow(log_edges, cmap='gray')
    plt.title('LoG Edge Detection')
    plt.tight_layout()  # remove overlap between titles and neighboring subplots

    output_path = os.path.join(os.path.dirname(__file__), "images", filename)
    plt.savefig(output_path)  # write the whole figure (all 3 techniques) to disk as a .jpg
    plt.show()


save_comparison(immg, 'Original Image', 'edge_detection_comparison_normal.jpg')
save_comparison(noisy_immg, 'Noisy Image', 'edge_detection_comparison_noisy.jpg')
save_comparison(varying_immg, 'Varying Intensity Image', 'edge_detection_comparison_varying_intensity.jpg')

plt.figure(figsize=(15, 5))  # wide canvas so a 1x3 row of 512x512 images is still readable
plt.subplot(131),plt.imshow(cv.cvtColor(immg, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(132),plt.imshow(cv.cvtColor(noisy_immg, cv.COLOR_BGR2RGB))
plt.title('Noisy Image')
plt.subplot(133),plt.imshow(cv.cvtColor(varying_immg, cv.COLOR_BGR2RGB))
plt.title('Varying Intensity Image')
plt.tight_layout()  # remove overlap between titles and neighboring subplots

variations_output_path = os.path.join(os.path.dirname(__file__), "images", "edge_detection_comparison_variations.jpg")
plt.savefig(variations_output_path)  # write just the 3 source image variations, no edge detection
plt.show()


# --- Low-contrast / noise degradation experiment ---
# I created a 256 x 256 grayscale image with a uniform background intensity of 30. The image
# contains exactly one filled square extending from pixel (35, 40) to (105, 110) with intensity
# 180 and one filled circle centered at (180, 165) with radius 36 and intensity 240.
#
# For the degraded condition, I changed the background, square, and circle intensities to 100,
# 125, and 180, respectively. The square-to-background contrast was therefore reduced to only 25
# gray levels. I then added zero-mean Gaussian noise with a standard deviation of 18 using a
# NumPy random-number generator with a fixed seed of 7. The fixed seed made the experiment
# reproducible.
#
# Before applying each detector, I used a 5 x 5 Gaussian filter with sigma = 1. Canny used
# cv2.Canny() with the L2 gradient option. Sobel combined 3 x 3 horizontal and vertical
# derivatives into a gradient magnitude, while Laplacian used the absolute response of a 3 x 3
# second-derivative kernel. Smoothing was important because differentiation amplifies
# high-frequency variation, particularly for a second-order operator such as Laplacian (Dey, 2025).
#
# The baseline thresholds were 50/150 for Canny hysteresis, 100 for Sobel magnitude, and 25 for
# Laplacian response. For the degraded image, I adjusted them to 35/105, 70, and 45. Lower Canny
# and Sobel thresholds were intended to recover the low-contrast square. I raised the Laplacian
# threshold because a lower value admitted excessive noise.

baseline_shapes = np.full((256, 256), 30, dtype=np.uint8)
cv.rectangle(baseline_shapes, (35, 40), (105, 110), 180, cv.FILLED)
cv.circle(baseline_shapes, (180, 165), 36, 240, cv.FILLED)

degraded_shapes = np.full((256, 256), 100, dtype=np.uint8)
cv.rectangle(degraded_shapes, (35, 40), (105, 110), 125, cv.FILLED)
cv.circle(degraded_shapes, (180, 165), 36, 180, cv.FILLED)

rng = np.random.default_rng(7)  # fixed seed so the noisy result is reproducible
degraded_noise = rng.normal(0, 18, degraded_shapes.shape)
degraded_shapes = np.clip(degraded_shapes.astype(np.float32) + degraded_noise, 0, 255).astype(np.uint8)


def detect_edges_shapes(gray, canny_thresholds, sobel_threshold, laplacian_threshold):
    blurred = cv.GaussianBlur(gray, (5, 5), 1)  # smoothing precedes differentiation to limit noise amplification
    canny_edges = cv.Canny(blurred, *canny_thresholds, L2gradient=True)

    sobel_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = cv.magnitude(sobel_x, sobel_y)
    sobel_edges = (sobel_magnitude > sobel_threshold).astype(np.uint8) * 255

    laplacian = np.abs(cv.Laplacian(blurred, cv.CV_64F, ksize=3))
    laplacian_edges = (laplacian > laplacian_threshold).astype(np.uint8) * 255

    return canny_edges, sobel_edges, laplacian_edges


def save_shapes_comparison(gray, label, filename, canny_thresholds, sobel_threshold, laplacian_threshold):
    canny_edges, sobel_edges, laplacian_edges = detect_edges_shapes(
        gray, canny_thresholds, sobel_threshold, laplacian_threshold
    )

    plt.figure(figsize=(20, 5))  # wide canvas so a 1x4 row of 256x256 images is still readable
    plt.subplot(141), plt.imshow(gray, cmap='gray')
    plt.title(label)
    plt.subplot(142), plt.imshow(canny_edges, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.subplot(143), plt.imshow(sobel_edges, cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.subplot(144), plt.imshow(laplacian_edges, cmap='gray')
    plt.title('Laplacian Edge Detection')
    plt.tight_layout()  # remove overlap between titles and neighboring subplots

    output_path = os.path.join(os.path.dirname(__file__), "images", filename)
    plt.savefig(output_path)  # write the whole figure (all 3 techniques) to disk as a .jpg
    plt.show()


save_shapes_comparison(baseline_shapes, 'Baseline Image', 'edge_detection_comparison_baseline_shapes.jpg', (50, 150), 100, 25)
save_shapes_comparison(degraded_shapes, 'Degraded Image', 'edge_detection_comparison_degraded_shapes.jpg', (35, 105), 70, 45)
