import cv2
import numpy as np

def crop_mpp_image(image_path, threshold_value=100):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    
    # Thresholding to isolate MPP from background
    _, thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour assuming it's the MPP region
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate the size of the square crop
    crop_size = min(w, h)
    
    # Round down the crop size to the nearest multiple of 10
    crop_size = (crop_size // 10) * 10
    
    # Calculate new coordinates for square cropping
    center_x, center_y = x + w // 2, y + h // 2
    new_x1 = max(center_x - crop_size // 2, 0)
    new_y1 = max(center_y - crop_size // 2, 0)
    new_x2 = min(center_x + crop_size // 2, image.shape[1])
    new_y2 = min(center_y + crop_size // 2, image.shape[0])

    # Crop the image to ensure it is a square
    cropped_image = image[new_y1:new_y2, new_x1:new_x2]

    # If the cropped image is not square, resize it
    if cropped_image.shape[0] != cropped_image.shape[1]:
        size = max(cropped_image.shape)  # Use the larger dimension to create a square
        square_image = np.zeros((size, size), dtype=np.uint8)  # Create a black square
        square_image[:cropped_image.shape[0], :cropped_image.shape[1]] = cropped_image  # Place the cropped image in the square
        return square_image

    return cropped_image
