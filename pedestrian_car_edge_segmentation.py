"""
Compare Canny's precise-but-gappy edges against a recall-oriented, morphologically-closed
edge map on a street scene with a pedestrian and a car. Which one fits better depends on the
task, not just the industry: forming a closed outline (segmentation/masking) favors recall and
F1, while pinpointing an exact boundary position (lane lines, a tripwire) favors Canny's
localization -- true for both self-driving cars and security cameras.
"""

import os  # standard library module for file/path operations
import cv2  # OpenCV library for computer vision tasks
import numpy as np  # used to stack the comparison images side by side

image_path = os.path.join(os.path.dirname(__file__), "images", "pedestrian_and_car.jpg")
image = cv2.imread(image_path)

if image is None:  # cv2.imread returns None if the file wasn't found or couldn't be read
    raise FileNotFoundError(f"Could not load image at: {image_path}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 1)  # smoothing precedes differentiation to limit noise amplification

# "Security camera" detector: Canny, tuned for sharp, precise localization of each edge pixel
security_edges = cv2.Canny(blurred, 80, 160, L2gradient=True)

# "Self-driving" detector: Sobel magnitude thresholded loosely, then morphologically closed so
# small contrast gaps along the pedestrian/car boundary don't leave the loop open
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))
_, sobel_binary = cv2.threshold(sobel_magnitude, 40, 255, cv2.THRESH_BINARY)  # low threshold favors recall over precision
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed_edges = cv2.morphologyEx(sobel_binary, cv2.MORPH_CLOSE, kernel, iterations=2)  # bridges small gaps into closed loops

security_contours, _ = cv2.findContours(security_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
driving_contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_area = 500  # drop tiny contours so noise specks aren't counted as detected objects
security_regions = [c for c in security_contours if cv2.contourArea(c) > min_area]
driving_regions = [c for c in driving_contours if cv2.contourArea(c) > min_area]

security_overlay = image.copy()
cv2.drawContours(security_overlay, security_regions, -1, (0, 0, 255), 2)

driving_overlay = image.copy()
cv2.drawContours(driving_overlay, driving_regions, -1, (0, 255, 0), 2)


def colorize_edges(edge_mask, color, background):
    colored = (background * 0.35).astype(np.uint8)  # dim the original photo so the colored edges pop against it
    colored[edge_mask > 0] = color
    return colored


security_edges_color = colorize_edges(security_edges, (0, 0, 255), image)  # red, matching the Canny contour color
driving_edges_color = colorize_edges(closed_edges, (0, 255, 0), image)  # green, matching the closed-contour color

print(f"Canny (precise localization): {len(security_regions)} closed region(s) above {min_area}px area")
print(f"Closed Sobel (closed contour): {len(driving_regions)} closed region(s) above {min_area}px area")

def label_panel(panel, text):
    labeled = panel.copy()
    band_height = max(40, labeled.shape[1] // 20)  # scale the label strip to the image's own width so it stays legible
    font_scale = labeled.shape[1] / 1200
    thickness = max(2, labeled.shape[1] // 700)
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], band_height), (0, 0, 0), cv2.FILLED)
    cv2.putText(labeled, text, (10, int(band_height * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return labeled


output_dir = os.path.join(os.path.dirname(__file__), "output_images")
os.makedirs(output_dir, exist_ok=True)

edges_comparison = np.hstack((
    label_panel(security_edges_color, "Canny: precise localization (lane lines, tripwires)"),
    label_panel(driving_edges_color, "Higher recall / contour completeness (vehicles, people, silhouettes)"),
))
contours_comparison = np.hstack((
    label_panel(security_overlay, "Canny contours: precise boundary position"),
    label_panel(driving_overlay, "Closed-contour segmentation: complete outline"),
))

edges_output_path = os.path.join(output_dir, "pedestrian_car_edge_maps.jpg")
contours_output_path = os.path.join(output_dir, "pedestrian_car_contour_segmentation.jpg")
cv2.imwrite(edges_output_path, edges_comparison)
cv2.imwrite(contours_output_path, contours_comparison)
print(f"Saved edge-map comparison to: {edges_output_path}")
print(f"Saved contour-segmentation comparison to: {contours_output_path}")

edges_window = "Canny: precise localization vs. Closed Sobel: closed contour"
contours_window = "Canny contours: boundary position vs. Closed-contour segmentation: complete outline"

cv2.namedWindow(edges_window, cv2.WINDOW_NORMAL)
cv2.imshow(edges_window, edges_comparison)
cv2.moveWindow(edges_window, 100, 100)

cv2.namedWindow(contours_window, cv2.WINDOW_NORMAL)
cv2.imshow(contours_window, contours_comparison)
cv2.moveWindow(contours_window, 150, 150)

window_names = [edges_window, contours_window]
while any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1 for name in window_names):  # keep waiting until both windows are closed
    if cv2.waitKey(50) != -1:  # exit early if a key is pressed while a window has focus
        break
cv2.destroyAllWindows()
