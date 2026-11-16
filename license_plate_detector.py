"""
Detect the Russian license plate in each car photo using OpenCV's pre-trained
haarcascade_russian_plate_number.xml classifier, running it on the grayscale
version of each image (cascades are trained on, and expect, single-channel
intensity data -- color adds nothing the classifier uses). A red box is drawn
around every detected region so the classifier's output can be checked visually.
us_car_1.jpg (a US plate) is included on purpose: it should produce NO
detection, showing the Russian-plate cascade doesn't just fire on any plate.
"""

import os  # standard library module for file/path operations
import cv2  # OpenCV library for computer vision tasks

IMAGE_NAMES = ["russian_car_1.jpg", "russian_car_2.jpg", "us_car_1.jpg"]

images_dir = os.path.join(os.path.dirname(__file__), "images")
output_dir = os.path.join(os.path.dirname(__file__), "output_images")
os.makedirs(output_dir, exist_ok=True)

cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
plate_cascade = cv2.CascadeClassifier(cascade_path)
if plate_cascade.empty():  # CascadeClassifier doesn't raise, it just loads nothing
    raise FileNotFoundError(f"Could not load cascade at: {cascade_path}")

results = []  # (name, color_image_with_box, plate_boxes) kept for the display loop at the end

for name in IMAGE_NAMES:
    image_path = os.path.join(images_dir, name)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # scaleFactor=1.05 (fine-grained size search) + minNeighbors=5 (require more overlapping
    # detections) found the plate cleanly on both Russian photos while producing zero
    # false positives on the US plate -- raw grayscale already works, so no extra
    # preprocessing (equalizeHist, blurring) was needed for these images.
    plate_boxes = plate_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)

    annotated = image.copy()
    for (x, y, w, h) in plate_boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)  # red boundary box (BGR)

    if len(plate_boxes) == 0:
        print(f"{name}: no license plate detected")
    else:
        for (x, y, w, h) in plate_boxes:
            print(f"{name}: plate detected at x={x}, y={y}, w={w}, h={h}")

    output_path = os.path.join(output_dir, f"{os.path.splitext(name)[0]}_plate_detected.jpg")
    cv2.imwrite(output_path, annotated)
    print(f"  saved annotated image to: {output_path}")

    results.append((name, annotated))

# Display every result so each detection (or lack of one) can be checked by eye
for index, (name, annotated) in enumerate(results):
    window_name = f"{name} - detected plate(s)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(window_name, annotated)
    cv2.moveWindow(window_name, 100 + index * 50, 100 + index * 50)

window_names = [f"{name} - detected plate(s)" for name, _ in results]
while any(cv2.getWindowProperty(w, cv2.WND_PROP_VISIBLE) >= 1 for w in window_names):
    if cv2.waitKey(50) != -1:
        break
cv2.destroyAllWindows()
