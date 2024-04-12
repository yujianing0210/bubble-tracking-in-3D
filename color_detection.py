import cv2
import numpy as np

# Load the image
image = cv2.imread('images/angle1.jpg')

# Convert the image from BGR to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for the red color
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
lower_red2 = np.array([350, 100, 100])
upper_red2 = np.array([360, 255, 255])

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

# Create a mask for the red color
mask1 = cv2.inRange(hsv, lower_red, upper_red)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)
mask3 = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Apply morphological operations to remove noise
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask3, cv2.MORPH_OPEN, kernel)

# Find contours of the red objects
contours, _ = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate over the contours and draw bounding rectangles around the red objects
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with detections
cv2.imshow('Red Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()