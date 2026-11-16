"""
Detect the face and eyes in my_image.jpg using Haar cascades, draw a red bounding box around
the eyes and a green circle around the face, then tag the image with the text "this is me".
"""

import os  # standard library module for file/path operations
import cv2  # OpenCV library for computer vision tasks

# 1. Import the image
image_path = os.path.join(os.path.dirname(__file__), "images", "my_image.jpg")  # full path to the image file
image = cv2.imread(image_path)  # read the image from disk into a NumPy array (BGR order)

if image is None:  # cv2.imread returns None if the file wasn't found or couldn't be read
    raise FileNotFoundError(f"Could not load image at: {image_path}")  # stop execution with a clear error

height, width = image.shape[:2]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Haar cascades run on grayscale intensity, not color

# 2. Load OpenCV's pre-trained Haar cascades and detect the face, then the eyes within it
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(100, 100))
if len(faces) == 0:
    raise RuntimeError("No face detected in the image.")

# Use the largest detected face in case of multiple/false-positive detections
face_x, face_y, face_w, face_h = max(faces, key=lambda f: f[2] * f[3])
print(f"Face box (pixels): top-left=({face_x}, {face_y}), width={face_w}, height={face_h}, "
      f"bottom-right=({face_x + face_w}, {face_y + face_h})")

# 3. Draw a green circle around the face, sized to cover from the top of the head to the chin
face_center = (face_x + face_w // 2, face_y + face_h // 2)
face_radius = int(max(face_w, face_h) * 0.65)
cv2.circle(image, face_center, face_radius, (0, 255, 0), thickness=8)  # BGR green

# 4. Detect eyes within the face region and draw one red bounding box around both of them
face_roi_gray = gray[face_y:face_y + face_h, face_x:face_x + face_w]
eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20))

if len(eyes) > 0:
    # Combine all detected eye boxes (relative to the face ROI) into one bounding box, offset back to full-image coords
    for i, (ex, ey, ew, eh) in enumerate(eyes):
        print(f"Eye {i} box (pixels, relative to face): top-left=({face_x + ex}, {face_y + ey}), "
              f"width={ew}, height={eh}")
    eye_lefts = [face_x + ex for (ex, ey, ew, eh) in eyes]
    eye_tops = [face_y + ey for (ex, ey, ew, eh) in eyes]
    eye_rights = [face_x + ex + ew for (ex, ey, ew, eh) in eyes]
    eye_bottoms = [face_y + ey + eh for (ex, ey, ew, eh) in eyes]
    pad = 15  # a little breathing room around the tightest fit
    eyes_top_left = (int(min(eye_lefts)) - pad, int(min(eye_tops)) - pad)
    eyes_bottom_right = (int(max(eye_rights)) + pad, int(max(eye_bottoms)) + pad)
else:
    print("No eyes detected; falling back to an estimated region.")
    # Fall back to the upper-middle portion of the face if the cascade can't find the eyes
    eyes_top_left = (face_x + int(face_w * 0.1), face_y + int(face_h * 0.25))
    eyes_bottom_right = (face_x + int(face_w * 0.9), face_y + int(face_h * 0.5))

print(f"Combined eye box (pixels): top-left={eyes_top_left}, bottom-right={eyes_bottom_right}")
cv2.rectangle(image, eyes_top_left, eyes_bottom_right, (0, 0, 255), thickness=8)  # BGR red

# 5. Tag the image with text, centered directly above the circle
font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 2.5, 6
text = "this is me"
(text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
text_x = face_center[0] - text_width // 2
text_y = face_center[1] - face_radius - 40  # 40px gap above the circle's top edge
# Draw a thick black outline first, then white on top, so the text stays readable on any background
cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 6, cv2.LINE_AA)
cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

# 6. Save the tagged image
output_path = os.path.join(os.path.dirname(__file__), "output_images", "my_image_tagged.jpg")
cv2.imwrite(output_path, image)
print(f"Saved tagged image to: {output_path}")

# 7. Display the result so it can be visually confirmed
cv2.namedWindow("Tagged Image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
display_height = 900  # scale the window to a reasonable on-screen size while keeping the full-res file untouched
display_width = int(width * display_height / height)
cv2.resizeWindow("Tagged Image", display_width, display_height)
cv2.imshow("Tagged Image", image)
cv2.moveWindow("Tagged Image", 100, 100)

while cv2.getWindowProperty("Tagged Image", cv2.WND_PROP_VISIBLE) >= 1:  # keep waiting until the window is closed
    if cv2.waitKey(50) != -1:  # exit early if a key is pressed while the window has focus
        break
cv2.destroyAllWindows()
