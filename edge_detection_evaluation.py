import os
import numpy as np
import cv2 as cv

# Loads the same source images edge_detection_comparison.py saves, so both scripts score identical pixels
images_dir = os.path.join(os.path.dirname(__file__), "images")

def load_image(filename):
    path = os.path.join(images_dir, filename)
    image = cv.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read {path}. Run edge_detection_comparison.py first to generate it.")
    return image

immg = load_image("edge_detection_source_normal.jpg")
noisy_immg = load_image("edge_detection_source_noisy.jpg")
varying_immg = load_image("edge_detection_source_varying_intensity.jpg")


def detect_edges(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    canny_edges = cv.Canny(image, 100, 200)
    sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv.convertScaleAbs(cv.magnitude(sobel_x, sobel_y))
    blurred = cv.GaussianBlur(gray, (3, 3), 0)
    log_edges = cv.convertScaleAbs(cv.Laplacian(blurred, cv.CV_64F, ksize=3))
    return canny_edges, sobel_edges, log_edges


def ground_truth_edges(shape):
    truth = np.zeros(shape[:2], dtype=np.uint8)  # true edges are known since the shapes were drawn, not photographed
    cv.rectangle(truth, (100, 100), (400, 400), 255, thickness=2)
    cv.circle(truth, (256, 256), 50, 255, thickness=2)
    return truth


def binarize(edge_map):
    _, binary = cv.threshold(edge_map, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # Sobel/LoG are grayscale, need a 0/255 mask to score
    return binary


def score_edges(detected, truth, alpha=1.0 / 9):
    detected_mask = detected > 0
    truth_mask = truth > 0
    tp = np.count_nonzero(detected_mask & truth_mask)
    fp = np.count_nonzero(detected_mask & ~truth_mask)
    fn = np.count_nonzero(~detected_mask & truth_mask)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # distance from every pixel to the nearest true-edge pixel, for Pratt's Figure of Merit
    dist_to_truth = cv.distanceTransform(255 - truth, cv.DIST_L2, 5)
    detected_dists = dist_to_truth[detected_mask]
    ideal_count = max(np.count_nonzero(truth_mask), np.count_nonzero(detected_mask))
    pfom = (1.0 / ideal_count) * np.sum(1.0 / (1.0 + alpha * detected_dists ** 2)) if ideal_count else 0.0

    return precision, recall, f1, pfom


def evaluate_edges(label, canny_edges, sobel_edges, log_edges, truth):
    detectors = {
        'Canny': canny_edges,
        'Sobel': binarize(sobel_edges),
        'LoG': binarize(log_edges),
    }
    print(f"\n{label} vs. ground-truth shape outlines")
    print(f"{'Detector':<8}{'Precision':>10}{'Recall':>10}{'F1':>8}{'PFOM':>8}")
    for name, detected in detectors.items():
        precision, recall, f1, pfom = score_edges(detected, truth)
        print(f"{name:<8}{precision:>10.3f}{recall:>10.3f}{f1:>8.3f}{pfom:>8.3f}")


def compute_snr_psnr(reference, test):
    reference = reference.astype(np.float64)
    test = test.astype(np.float64)
    mse = np.mean((reference - test) ** 2)
    signal_power = np.mean(reference ** 2)
    snr = 10 * np.log10(signal_power / mse) if mse else float('inf')
    psnr = 10 * np.log10((255.0 ** 2) / mse) if mse else float('inf')
    return snr, psnr


ground_truth = ground_truth_edges(immg.shape)

for label, image in [
    ('Original Image', immg),
    ('Noisy Image', noisy_immg),
    ('Varying Intensity Image', varying_immg),
]:
    canny_edges, sobel_edges, log_edges = detect_edges(image)
    evaluate_edges(label, canny_edges, sobel_edges, log_edges, ground_truth)

print("\nImage-level degradation (whole image vs. clean original, not edge-specific)")
for label, variant in [('Noisy Image', noisy_immg), ('Varying Intensity Image', varying_immg)]:
    snr, psnr = compute_snr_psnr(immg, variant)
    print(f"{label:<24}SNR={snr:6.2f} dB  PSNR={psnr:6.2f} dB")
