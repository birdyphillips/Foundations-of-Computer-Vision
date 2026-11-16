"""
CSC515 Portfolio Milestone
Import a banknote image into OpenCV and examine its pixel matrix.
"""

import os  # standard library module for file/path operations
# import ctypes  # was used to force the OpenCV window into OS focus; commented out to keep things simple
import cv2  # OpenCV library for computer vision tasks
import numpy as np  # numerical library used to inspect the pixel array

# try:
#     ctypes.windll.shcore.SetProcessDpiAwareness(1)  # avoid DPI virtualization so the window can be dragged normally
# except AttributeError:
#     ctypes.windll.user32.SetProcessDPIAware()  # fallback for older Windows versions

# 1. Import the image
image_path = os.path.join(os.path.dirname(__file__), "images", "banknotes_image.jpg")  # build full path to the image file
image = cv2.imread(image_path)  # read the image from disk into a NumPy array (BGR order)

if image is None:  # cv2.imread returns None if the file wasn't found or couldn't be read
    raise FileNotFoundError(f"Could not load image at: {image_path}")  # stop execution with a clear error

# 2. Examine the pixel matrix
height, width, channels = image.shape  # dimensions of the pixel matrix
print(f"Image shape (height, width, channels): {image.shape}")
print(f"Data type of pixel values: {image.dtype}")
print(f"Total number of pixels: {height * width}")
print(f"Minimum pixel value: {image.min()}, Maximum pixel value: {image.max()}")

# Print the full matrix for the region containing the "100" numeral on the right, upside-down bill
# (two bills are clipped to the line; the right one clearly shows its orange "100" numeral),
# since that's a more meaningful sample than an arbitrary corner block
numeral_row_start, numeral_row_end = 4, 33
numeral_col_start, numeral_col_end = 62, 106
sample_region = image[numeral_row_start:numeral_row_end, numeral_col_start:numeral_col_end]
print(
    f"\nOriginal image - '100' numeral region "
    f"(rows {numeral_row_start}-{numeral_row_end}, cols {numeral_col_start}-{numeral_col_end}) BGR values:"
)
print(sample_region)

# Print the BGR value of a pixel at the center of the "100" numeral region
numeral_center_row = (numeral_row_start + numeral_row_end) // 2
numeral_center_col = (numeral_col_start + numeral_col_end) // 2
center_pixel = image[numeral_center_row, numeral_center_col]
print(f"\nCenter pixel of the '100' numeral (row={numeral_center_row}, col={numeral_center_col}) BGR value: {center_pixel}")

# The bill hangs upside down, so rotate the numeral crop 180 degrees to read it right-side up as "100"
numeral_upright = cv2.rotate(sample_region, cv2.ROTATE_180)
numeral_upright_path = os.path.join(os.path.dirname(__file__), "banknotes_100_numeral_upright.jpg")
cv2.imwrite(numeral_upright_path, numeral_upright)
print(f"\nSaved upright '100' numeral crop to: {numeral_upright_path}")

# Isolate the portrait (the man's face) on the left, upside-down bill -- it's partly cut off by the
# photo's frame and the clothespin, so this crop captures the visible portion of the portrait
face_row_start, face_row_end = 45, 83
face_col_start, face_col_end = 0, 35
face_region = image[face_row_start:face_row_end, face_col_start:face_col_end]
print(
    f"\nOriginal image - left bill's face/portrait region "
    f"(rows {face_row_start}-{face_row_end}, cols {face_col_start}-{face_col_end}) BGR values:"
)
print(face_region)

# Rotate the face crop 90 degrees right (clockwise) so the portrait reads right-side up
face_upright = cv2.rotate(face_region, cv2.ROTATE_90_CLOCKWISE)

# Upscale with cubic interpolation so the saved image looks smoother instead of blocky/pixelated
face_upright_smooth = cv2.resize(face_upright, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
face_upright_path = os.path.join(os.path.dirname(__file__), "banknotes_face_upright.jpg")
cv2.imwrite(face_upright_path, face_upright_smooth)
print(f"\nSaved upright face/portrait crop to: {face_upright_path}")

# 3. Build an explicit rotation matrix and apply it so the transformation matrix can be discussed
rotation_angle_degrees = 90
rotation_center = (width / 2, height / 2)
rotation_matrix = cv2.getRotationMatrix2D(rotation_center, rotation_angle_degrees, scale=1.0)

# Expand the output canvas so the non-square image isn't cropped after rotation
cos = abs(rotation_matrix[0, 0])
sin = abs(rotation_matrix[0, 1])
new_width = int((height * sin) + (width * cos))
new_height = int((height * cos) + (width * sin))
rotation_matrix[0, 2] += (new_width / 2) - rotation_center[0]
rotation_matrix[1, 2] += (new_height / 2) - rotation_center[1]

print(f"\n2x3 affine rotation matrix ({rotation_angle_degrees} degrees about {rotation_center}):")
print(rotation_matrix)

rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

# Print the rotated image's top-left 5x5 block so it can be compared to the original block
rotated_sample_region = rotated_image[0:5, 0:5]
print("\nRotated image (90 degrees) - Top-left 5x5 pixel block (BGR values):")
print(rotated_sample_region)

# Save the transformed image to disk so it can be attached to the assignment post
rotated_image_path = os.path.join(os.path.dirname(__file__), "banknotes_image_rotated.jpg")
cv2.imwrite(rotated_image_path, rotated_image)
print(f"\nSaved rotated image to: {rotated_image_path}")

# 4. Build an explicit translation matrix and apply it so pixels are shifted within the canvas
shift_x, shift_y = 20, 15  # pixels to shift right and down
translation_matrix = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)
print(f"\n2x3 affine translation matrix (shift_x={shift_x}, shift_y={shift_y}):")
print(translation_matrix)

translated_image = cv2.warpAffine(image, translation_matrix, (width, height))

# Print the translated image's top-left 5x5 block so it can be compared to the original block
translated_sample_region = translated_image[0:5, 0:5]
print("\nTranslated image - Top-left 5x5 pixel block (BGR values):")
print(translated_sample_region)

# Save the translated image to disk so it can be attached to the assignment post
translated_image_path = os.path.join(os.path.dirname(__file__), "banknotes_image_translated.jpg")
cv2.imwrite(translated_image_path, translated_image)
print(f"\nSaved translated image to: {translated_image_path}")

# 5. Display the original, rotated, and translated images at their original size so the pixel data can be visually confirmed
display_image = image
display_rotated = rotated_image
display_translated = translated_image

cv2.namedWindow("Banknotes Image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # keep aspect ratio while resizing
cv2.imshow("Banknotes Image", display_image)
cv2.moveWindow("Banknotes Image", 100, 100)  # place it at a known, visible screen position

cv2.namedWindow("Banknotes Image (Rotated 90)", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # keep aspect ratio while resizing
cv2.imshow("Banknotes Image (Rotated 90)", display_rotated)
cv2.moveWindow("Banknotes Image (Rotated 90)", 100 + display_image.shape[1] + 40, 100)  # place beside the original

cv2.namedWindow("Banknotes Image (Translated)", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # keep aspect ratio while resizing
cv2.imshow("Banknotes Image (Translated)", display_translated)
cv2.moveWindow(
    "Banknotes Image (Translated)", 100 + display_image.shape[1] + display_rotated.shape[1] + 80, 100
)  # place beside the rotated window

cv2.namedWindow("100 Numeral (Upright)", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # keep aspect ratio while resizing
cv2.imshow("100 Numeral (Upright)", numeral_upright)
cv2.moveWindow(
    "100 Numeral (Upright)",
    100 + display_image.shape[1] + display_rotated.shape[1] + display_translated.shape[1] + 120,
    100,
)  # place beside the translated window

cv2.namedWindow("Face (Upright)", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # keep aspect ratio while resizing
cv2.imshow("Face (Upright)", face_upright_smooth)
cv2.moveWindow(
    "Face (Upright)",
    100 + display_image.shape[1] + display_rotated.shape[1] + display_translated.shape[1] + numeral_upright.shape[1] + 140,
    100,
)  # place beside the numeral window

while (
    cv2.getWindowProperty("Banknotes Image", cv2.WND_PROP_VISIBLE) >= 1
    or cv2.getWindowProperty("Banknotes Image (Rotated 90)", cv2.WND_PROP_VISIBLE) >= 1
    or cv2.getWindowProperty("Banknotes Image (Translated)", cv2.WND_PROP_VISIBLE) >= 1
    or cv2.getWindowProperty("100 Numeral (Upright)", cv2.WND_PROP_VISIBLE) >= 1
    or cv2.getWindowProperty("Face (Upright)", cv2.WND_PROP_VISIBLE) >= 1
):  # keep waiting until every window has been closed
    if cv2.waitKey(50) != -1:  # exit early if a key is pressed while a window has focus
        break
# hwnd = ctypes.windll.user32.FindWindowW(None, "Banknotes Image")  # locate the window by its title
# if hwnd:
#     ctypes.windll.user32.SetForegroundWindow(hwnd)  # bring it to the foreground so it's draggable right away
# while (
#     cv2.getWindowProperty("Banknotes Image", cv2.WND_PROP_VISIBLE) >= 1
#     and cv2.getWindowProperty("Banknotes Image (Rotated 90)", cv2.WND_PROP_VISIBLE) >= 1
#     and cv2.getWindowProperty("Banknotes Image (Translated)", cv2.WND_PROP_VISIBLE) >= 1
# ):  # keep waiting only while all windows are open
#     if cv2.waitKey(50) != -1:  # exit early if a key is pressed while a window has focus
#         break
cv2.destroyAllWindows()
