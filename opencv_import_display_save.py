"""
CSC515 Module 1 Portfolio Milestone
Import, display, and save a copy of an image using OpenCV.
"""

import os  # standard library module for file/path operations
import ctypes  # used to force the OpenCV window into OS focus so it can be dragged immediately
import cv2  # OpenCV library for computer vision tasks

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # avoid DPI virtualization so the window can be dragged normally
except AttributeError:
    ctypes.windll.user32.SetProcessDPIAware()  # fallback for older Windows versions

# 1. Import the image
# The image file lives in the same folder as this script.
image_path = os.path.join(os.path.dirname(__file__), "brain_image.jpg")  # build full path to the image file
image = cv2.imread(image_path)  # read the image from disk into a NumPy array

if image is None:  # cv2.imread returns None if the file wasn't found or couldn't be read
    raise FileNotFoundError(f"Could not load image at: {image_path}")  # stop execution with a clear error

# 2. Display the image
cv2.namedWindow("Brain Image", cv2.WINDOW_NORMAL)  # normal window chrome so it can be dragged/resized
cv2.imshow("Brain Image", image)  # open a window and show the image
cv2.moveWindow("Brain Image", 100, 100)  # place it at a known, visible screen position
cv2.waitKey(1)  # let the window finish its first paint before we try to focus it
hwnd = ctypes.windll.user32.FindWindowW(None, "Brain Image")  # locate the window by its title
if hwnd:
    ctypes.windll.user32.SetForegroundWindow(hwnd)  # bring it to the foreground so it's draggable right away
cv2.waitKey(0)  # wait indefinitely for a key press so the window stays open
cv2.destroyAllWindows()  # close all OpenCV windows once a key is pressed

# 3. Write a copy of the image to a directory on the desktop
desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop", "CSC515_Images")  # target folder on the desktop
os.makedirs(desktop_dir, exist_ok=True)  # create the folder if it doesn't already exist

output_path = os.path.join(desktop_dir, "brain_image_copy.jpg")  # full path for the saved copy
cv2.imwrite(output_path, image)  # write the image data to the new file

print(f"Image copy saved to: {output_path}")  # confirm where the copy was saved
