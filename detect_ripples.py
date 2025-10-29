import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# Usage: python detect_ripples.py path/to/lake_image.jpg

if len(sys.argv) < 2:
    print("Usage: python detect_ripples.py path/to/lake_image.jpg")
    sys.exit()

image_path = sys.argv[1]

# Load image
image = cv2.imread(image_path)
if image is None:
    print("Image not found. Check the file name and path.")
    sys.exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blur, 50, 150)

# Optional: detect circles (ripples are roughly circular)
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=10, maxRadius=200)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

# Display
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Detected Edges')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Ripples Detected')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
