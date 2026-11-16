"""
Crop the portrait (the man's face) off the left, upside-down $100 bill, rotate it right-side up,
and smooth the tiny crop so it isn't blocky/pixelated when shared or viewed at a larger size.
"""

import os  # standard library module for file/path operations
import cv2  # OpenCV library for computer vision tasks

# 1. Import the image
image_path = os.path.join(os.path.dirname(__file__), "images", "banknotes_image.jpg")  # full path to the image file
image = cv2.imread(image_path)  # read the image from disk into a NumPy array (BGR order)

if image is None:  # cv2.imread returns None if the file wasn't found or couldn't be read
    raise FileNotFoundError(f"Could not load image at: {image_path}")  # stop execution with a clear error

# 2. Isolate the portrait (the man's face) on the left, upside-down bill -- it's partly cut off by the
# photo's frame and the clothespin, so this crop captures the visible portion of the portrait
face_row_start, face_row_end = 45, 83
face_col_start, face_col_end = 0, 35
face_region = image[face_row_start:face_row_end, face_col_start:face_col_end]
print(
    f"Original image - left bill's face/portrait region "
    f"(rows {face_row_start}-{face_row_end}, cols {face_col_start}-{face_col_end}) BGR values:"
)
print(face_region)

# 3. Rotate the face crop 90 degrees right (clockwise) so the portrait reads right-side up
face_upright = cv2.rotate(face_region, cv2.ROTATE_90_CLOCKWISE)

# 4. Upscale with cubic interpolation so the saved image looks smoother instead of blocky/pixelated
face_upright_smooth = cv2.resize(face_upright, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
face_upright_path = os.path.join(os.path.dirname(__file__), "banknotes_face_upright.jpg")
cv2.imwrite(face_upright_path, face_upright_smooth)
print(f"\nSaved upright face/portrait crop to: {face_upright_path}")

# 5. Display the result so it can be visually confirmed
cv2.namedWindow("Face (Upright)", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # keep aspect ratio while resizing
cv2.imshow("Face (Upright)", face_upright_smooth)
cv2.moveWindow("Face (Upright)", 100, 100)

while cv2.getWindowProperty("Face (Upright)", cv2.WND_PROP_VISIBLE) >= 1:  # keep waiting until the window is closed
    if cv2.waitKey(50) != -1:  # exit early if a key is pressed while the window has focus
        break
cv2.destroyAllWindows()
