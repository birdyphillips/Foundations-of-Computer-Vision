"""
Take the plate region the cascade found in license_plate_detector.py,
crop it out, and prepare ONLY that crop for character recognition:
  1. deskew it (small camera-angle tilt makes characters lean, which hurts OCR)
     by fitting a minimum-area rectangle to the plate's own border contour and
     rotating by that angle so the plate is level;
  2. upscale it (cascade crops are tiny, often <150px wide -- Tesseract reads
     bigger glyphs far more reliably);
  3. Otsu-threshold it to pure black/white so the characters stand out from the
     plate background.
Requires the Tesseract-OCR binary (not just the pytesseract pip package).
Install it with:  winget install --id UB-Mannheim.TesseractOCR -e
"""

import os  # standard library module for file/path operations
import shutil  # used to locate the tesseract binary on PATH
import cv2  # OpenCV library for computer vision tasks
import pytesseract

# Russian plates only -- us_car_1.jpg had no plate detected, so there is nothing to align/read
IMAGE_NAMES = ["russian_car_1.jpg", "russian_car_2.jpg"]

images_dir = os.path.join(os.path.dirname(__file__), "images")
output_dir = os.path.join(os.path.dirname(__file__), "output_images")
os.makedirs(output_dir, exist_ok=True)

# UB-Mannheim's Windows installer doesn't always add tesseract.exe to PATH, so fall back to
# its default install location if `tesseract` isn't already found
if shutil.which("tesseract") is None:
    default_install = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_install):
        pytesseract.pytesseract.tesseract_cmd = default_install

cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
plate_cascade = cv2.CascadeClassifier(cascade_path)
if plate_cascade.empty():
    raise FileNotFoundError(f"Could not load cascade at: {cascade_path}")

# Russian plate characters are limited to digits plus the Cyrillic letters that happen to be
# shaped like Latin letters (A,B,E,K,M,H,O,P,C,T,Y,X); constraining Tesseract's English model
# to just this whitelist avoids it guessing unrelated punctuation/symbols
PLATE_CHAR_WHITELIST = "ABEKMHOPCTYX0123456789"

results = []  # (name, deskewed_plate, ocr_text) kept for the display loop at the end

for name in IMAGE_NAMES:
    image_path = os.path.join(images_dir, name)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plate_boxes = plate_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
    if len(plate_boxes) == 0:
        print(f"{name}: no plate detected, skipping")
        continue

    x, y, w, h = plate_boxes[0]
    # pad around the cascade box so the plate's full border is included for the angle fit,
    # without pulling in so much of the bumper that it changes the detected rectangle
    pad_x, pad_y = int(w * 0.1), int(h * 0.3)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(gray.shape[1], x + w + pad_x), min(gray.shape[0], y + h + pad_y)
    plate_crop = gray[y0:y1, x0:x1]

    # 1. Deskew: fit the plate's own rectangular border with minAreaRect to read off its tilt
    _, plate_mask = cv2.threshold(plate_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(plate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    (_, _), (rect_w, rect_h), angle = cv2.minAreaRect(largest_contour)
    if rect_w < rect_h:  # minAreaRect's angle is only meaningful relative to its own w/h axes
        angle -= 90

    center = (plate_crop.shape[1] / 2, plate_crop.shape[0] / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        plate_crop, rotation_matrix, (plate_crop.shape[1], plate_crop.shape[0]),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )

    # 2. Scale up so individual characters are large enough for Tesseract to resolve
    scaled = cv2.resize(deskewed, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # 3. Binarize so characters are pure black/white, which is what Tesseract is tuned for
    _, ocr_ready = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    output_path = os.path.join(output_dir, f"{os.path.splitext(name)[0]}_plate_aligned.jpg")
    cv2.imwrite(output_path, ocr_ready)
    print(f"{name}: deskewed by {angle:.1f} deg, saved aligned plate to: {output_path}")

    try:
        # --psm 7 treats the crop as a single line of text, which matches a license plate
        ocr_text = pytesseract.image_to_string(
            ocr_ready, config=f"--psm 7 -c tessedit_char_whitelist={PLATE_CHAR_WHITELIST}"
        ).strip()
        print(f"{name}: recognized text -> {ocr_text!r}")
    except pytesseract.TesseractNotFoundError:
        ocr_text = "(Tesseract-OCR not installed -- run: winget install --id UB-Mannheim.TesseractOCR -e)"
        print(f"{name}: {ocr_text}")

    results.append((name, ocr_ready, ocr_text))

# Display every aligned plate so the deskew/scale/threshold result can be checked by eye
for index, (name, ocr_ready, _) in enumerate(results):
    window_name = f"{name} - aligned plate"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(window_name, ocr_ready)
    cv2.moveWindow(window_name, 100 + index * 50, 100 + index * 50)

window_names = [f"{name} - aligned plate" for name, _, _ in results]
while any(cv2.getWindowProperty(w, cv2.WND_PROP_VISIBLE) >= 1 for w in window_names):
    if cv2.waitKey(50) != -1:
        break
cv2.destroyAllWindows()
