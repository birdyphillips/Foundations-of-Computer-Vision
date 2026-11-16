"""
Detect + align + OCR license plates in one file.

Tries the Russian Haar cascade first, falls back to an edge/contour-based
detector for plates it misses (e.g. US), then deskews, upscales, and
Otsu-thresholds the crop before running Tesseract OCR on it.

Requires the Tesseract-OCR binary (not just the pytesseract pip package).
Install it with:  winget install --id UB-Mannheim.TesseractOCR -e
"""

import os  # path/filesystem helpers
import shutil  # locates tesseract.exe on PATH
import cv2
import numpy as np  # kernel math and contour point arrays
import pytesseract
from matplotlib import pyplot as plt  # renders the 3-column result grids

IMAGE_NAMES = ["russian_car_1.jpg", "russian_car_2.jpg", "us_car_1.jpg"]

images_dir = os.path.join(os.path.dirname(__file__), "images")
output_dir = os.path.join(os.path.dirname(__file__), "output_images")
os.makedirs(output_dir, exist_ok=True)

# UB-Mannheim's installer often skips adding tesseract.exe to PATH; check its default folder next
if shutil.which("tesseract") is None:
    default_install = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_install):
        pytesseract.pytesseract.tesseract_cmd = default_install

cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
plate_cascade = cv2.CascadeClassifier(cascade_path)
if plate_cascade.empty():  # a bad path silently loads an empty classifier rather than raising
    raise FileNotFoundError(f"Could not load cascade at: {cascade_path}")

# Only digits plus the Cyrillic letters that visually match Latin ones appear on these plates;
# limiting Tesseract to them keeps it from mistaking noise for stray punctuation
PLATE_CHAR_WHITELIST = "ABEKMHOPCTYX0123456789"
# the fallback detector targets non-Russian plates, which draw from the full Latin alphabet
GENERIC_PLATE_CHAR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def detect_plate_generic(gray):
    """Locates a plate-shaped candidate using geometry and edge texture, with no trained model."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    _, edges = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # a wide, short kernel fuses neighboring character strokes into a single blob
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_score = -1
    img_h, img_w = gray.shape
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area_fraction = (w * h) / float(img_w * img_h)
        y_center_fraction = (y + h / 2.0) / float(img_h)
        mean_brightness = gray[y:y + h, x:x + w].mean()
        # filters out regions that don't match a plate's typical proportions and placement
        if (2.0 <= aspect_ratio <= 6.0 and 0.001 <= area_fraction <= 0.06
                and 0.45 <= y_center_fraction <= 0.95 and mean_brightness >= 110):
            edge_density = cv2.countNonZero(edges[y:y + h, x:x + w]) / float(w * h)  # denser edges imply more characters
            if edge_density > best_score:
                best_score = edge_density
                best_box = (x, y, w, h)

    return best_box


original_results = []  # holds (name, image) pairs for the first comparison grid
detection_results = []  # holds (name, boxed_image) pairs for the second grid
ocr_results = []  # holds (label, crop, recognized_text) per plate found
image_summary = {}  # keyed by image name, feeds the printed results table

for name in IMAGE_NAMES:
    image_path = os.path.join(images_dir, name)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")
    original_results.append((name, image))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # a fine scaleFactor plus a stricter minNeighbors gave clean hits on both Russian photos
    # with no stray detections on the US plate during testing
    plate_boxes = plate_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
    used_generic_detector = False

    if len(plate_boxes) == 0:
        # the classifier only knows the Russian template, so try the geometric method next
        generic_box = detect_plate_generic(gray)
        if generic_box is not None:
            plate_boxes = [generic_box]
            used_generic_detector = True

    annotated = image.copy()
    box_color = (0, 255, 255) if used_generic_detector else (0, 255, 0)  # yellow marks the fallback, green marks the cascade
    for (x, y, w, h) in plate_boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 3)

    if len(plate_boxes) == 0:
        image_summary[name] = {"method": "not detected"}
    else:
        method = "generic detector" if used_generic_detector else "Russian cascade"
        image_summary[name] = {"method": f"{method} ({len(plate_boxes)})"}

    detected_path = os.path.join(output_dir, f"{os.path.splitext(name)[0]}_plate_detected.jpg")
    cv2.imwrite(detected_path, annotated)

    # one pixel from the image center keeps the results table narrow enough to screenshot
    img_h, img_w = annotated.shape[:2]
    b, g, r = (int(c) for c in annotated[img_h // 2, img_w // 2])
    image_summary[name]["center_pixel"] = f"({b},{g},{r})"

    detection_results.append((name, annotated))

    if len(plate_boxes) == 0:
        continue  # no box means no region exists yet to deskew or read

    for plate_index, (x, y, w, h) in enumerate(plate_boxes):
        # numbers the plate only when a photo contains more than one, like the 2-car Russian scene
        label = f"{name} #{plate_index + 1}" if len(plate_boxes) > 1 else name

        # extends the box slightly to capture the plate's full border for angle fitting
        pad_x, pad_y = int(w * 0.1), int(h * 0.3)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(gray.shape[1], x + w + pad_x), min(gray.shape[0], y + h + pad_y)
        plate_crop = gray[y0:y1, x0:x1]

        # step 1: minAreaRect reads the tilt directly off the plate's own border contour
        _, plate_mask = cv2.threshold(plate_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(plate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        rect_center, (rect_w, rect_h), raw_angle = cv2.minAreaRect(largest_contour)
        angle = raw_angle
        if rect_w < rect_h:  # the reported angle only makes sense relative to the rect's own axes
            angle -= 90

        center = (plate_crop.shape[1] / 2, plate_crop.shape[0] / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(
            plate_crop, rotation_matrix, (plate_crop.shape[1], plate_crop.shape[0]),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )

        # trims back down to just the plate itself; the extra padding would otherwise confuse
        # Tesseract with bright bumper or background pixels surrounding the characters
        box_points = cv2.boxPoints((rect_center, (rect_w, rect_h), raw_angle))
        rotated_box_points = cv2.transform(np.array([box_points]), rotation_matrix)[0]
        bx0 = max(0, int(rotated_box_points[:, 0].min()))
        by0 = max(0, int(rotated_box_points[:, 1].min()))
        bx1 = min(deskewed.shape[1], int(rotated_box_points[:, 0].max()))
        by1 = min(deskewed.shape[0], int(rotated_box_points[:, 1].max()))
        if bx1 > bx0 and by1 > by0:
            deskewed = deskewed[by0:by1, bx0:bx1]

        # step 2: enlarging the crop gives Tesseract bigger glyphs to work with
        scaled = cv2.resize(deskewed, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        # step 3: Tesseract performs best on clean black-and-white input, not grayscale
        _, ocr_ready = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        aligned_path = os.path.join(output_dir, f"{os.path.splitext(name)[0]}_plate_{plate_index + 1}_aligned.jpg")
        cv2.imwrite(aligned_path, ocr_ready)

        try:
            # --psm 7 tells Tesseract to expect one line of text, matching a plate's layout
            whitelist = GENERIC_PLATE_CHAR_WHITELIST if used_generic_detector else PLATE_CHAR_WHITELIST
            ocr_text = pytesseract.image_to_string(
                ocr_ready, config=f"--psm 7 -c tessedit_char_whitelist={whitelist}"
            ).strip()
        except pytesseract.TesseractNotFoundError:
            ocr_text = "(Tesseract not installed)"

        ocr_results.append((label, ocr_ready, ocr_text))

# a 3-column layout (one image per column) keeps the whole summary within one screenshot
col_width = 22
print("\n" + " | ".join(name[:col_width].ljust(col_width) for name in IMAGE_NAMES))
print("-" * (col_width * len(IMAGE_NAMES) + 3 * (len(IMAGE_NAMES) - 1)))
print(" | ".join(image_summary[name]["method"][:col_width].ljust(col_width) for name in IMAGE_NAMES))
print(" | ".join(image_summary[name]["center_pixel"][:col_width].ljust(col_width) for name in IMAGE_NAMES))
for row in range(3):  # covers the max plates seen in one photo (3, from the multi-car scene)
    row_cells = []
    for name in IMAGE_NAMES:
        matches = [text for result_name, _, text in ocr_results
                   if result_name == name or result_name.startswith(f"{name} #")]
        text = matches[row] if row < len(matches) else ""
        row_cells.append((text or "-")[:col_width].ljust(col_width))
    if any(cell.strip() not in ("", "-") for cell in row_cells):
        print(" | ".join(row_cells))
print()

def save_grid(results, window_title, output_name, cmap=None):
    """Arranges each result in its own labeled column and writes the figure to output_images/."""
    cols = len(results)
    fig = plt.figure(window_title, figsize=(cols * 4, 4))
    for i, (name, img) in enumerate(results):
        plt.subplot(1, cols, i + 1)
        # without fixed bounds, a flat-value placeholder gets normalized down to solid black
        plt.imshow(img if cmap else cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap=cmap, vmin=0, vmax=255 if cmap else None)
        plt.title(name)
        plt.axis("off")
    fig.tight_layout()

    output_path = os.path.join(output_dir, output_name)
    fig.savefig(output_path, dpi=150)
    print(f"Saved {window_title} grid to: {output_path}")


# some images contribute more than one crop (the 2-plate scene), others contribute none,
# so a filler image stands in wherever no plate was ever isolated
aligned_results = [(label, crop) for label, crop, _ in ocr_results]
for name in IMAGE_NAMES:
    if not any(label == name or label.startswith(f"{name} #") for label, _ in aligned_results):
        placeholder = np.full((100, 300), 128, dtype=np.uint8)  # flat gray stand-in
        aligned_results.append((name, placeholder))

# three side-by-side summaries: raw photos, boxed detections, and the OCR input crops
save_grid(original_results, "Original images", "original_images_grid.jpg")
save_grid(detection_results, "Detected license plates", "detected_plates_grid.jpg")
save_grid(aligned_results, "License plate numbers (OCR input)", "aligned_plates_grid.jpg", cmap="gray")
plt.show()
