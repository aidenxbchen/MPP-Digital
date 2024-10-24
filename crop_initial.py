import cv2
import numpy as np

# Step 1: Load and resize the MPP image
image = cv2.imread('./Image/241015/1-1-2.jpg', cv2.IMREAD_GRAYSCALE)
print(image.shape)
# image = cv2.resize(image, (600, 800))

cropped_init_image = image[2200:5800, 1000:4600]

 # Apply threshold using the current trackbar value
_, thresholded = cv2.threshold(image, 83, 255, cv2.THRESH_BINARY)
    
# Find contours again after applying the threshold
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
if contours:
 # Find the largest contour assuming it's the MPP region
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
        
    # Make sure the bounding box is square
    max_side = max(w, h)  # Find the largest side
        
    # Calculate adjustments to make the bounding box square
    new_x = x
    new_y = y
    if w > h:
        new_y = y - (w - h) // 2  # Adjust y to center vertically
        new_y = max(new_y, 0)  # Make sure it stays within the image
        h = w  # Make the height equal to the width
    else:
        new_x = x - (h - w) // 2  # Adjust x to center horizontally
        new_x = max(new_x, 0)  # Make sure it stays within the image
        w = h  # Make the width equal to the height

    # Ensure the square fits within the image boundaries
    new_x = min(new_x, image.shape[1] - w)
    new_y = min(new_y, image.shape[0] - h)
        
    # Crop the image to the square bounding box
    cropped_image = image[new_y:new_y+h, new_x:new_x+w]
else:
    cropped_image = image.copy()  # If no contour is found, just display the original image

# Display the updated images
cv2.imwrite('./Image/241015/1-1-2-cropped.jpg',cropped_image)
print(f"Cropped image size: {cropped_image.shape}")  # Output the cropped image size

# # Create a window for displaying the results
# cv2.imshow('Raw Image', image)
# cv2.imshow('Cropped Image', cropped_init_image)

# # Wait indefinitely for a key press, then close all windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()