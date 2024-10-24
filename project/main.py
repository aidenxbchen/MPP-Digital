from image_cropping import crop_mpp_image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and Crop the MPP Image
image_path = '/home/aiden/MPP_digital/Image/241012-test/image1.jpg'  # Use the absolute path
cropped_image = crop_mpp_image(image_path, threshold_value=100)

print(cropped_image.shape)

# Step 2: Generate First Image with Overall Average Brightness
overall_avg_brightness = np.mean(cropped_image)
first_image = cropped_image.copy()
cv2.putText(first_image, f"Avg Brightness: {overall_avg_brightness:.1f}", 
            (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 255, 0), 2)

# Step 3: Calculate Pixels per cm² (Resolution)
# Assume the real MPP size is 10x10 cm, so we calculate how many pixels equal 1 cm
real_mpp_size_cm = 10  # in cm
height, width = cropped_image.shape
pixels_per_cm = width / real_mpp_size_cm  # Should be the same for height since it's square

# Divide the cropped image into 1x1 cm² regions
region_size = int(pixels_per_cm)  # Number of pixels per cm region

# Initialize arrays to store average brightness for each subregion
avg_brightness_per_region = []

for i in range(0, height, region_size):
    row_brightness = []
    for j in range(0, width, region_size):
        # Extract the region (1x1 cm region)
        region = cropped_image[i:i+region_size, j:j+region_size]
        
        # Compute the average brightness of this region
        avg_brightness = np.mean(region)
        row_brightness.append(avg_brightness)
        
        # Step 4: Annotate this region with brightness (for output image 2)
        cv2.putText(cropped_image, f"{avg_brightness:.1f}", (j + 5, i + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 1)
        
    avg_brightness_per_region.append(row_brightness)

# Step 5: Output the Images
# Save the first image to disk
plt.imshow(first_image, cmap='gray', aspect='equal')
plt.title('Cropped Image with Overall Average Brightness')
plt.savefig('/home/aiden/MPP_digital/project/first_image.png')  # Save instead of show

# Save the second image to disk
plt.imshow(cropped_image, cmap='gray', aspect='equal')
plt.title('Cropped Image with Subregion Brightness')
plt.savefig('/home/aiden/MPP_digital/project/second_image.png')  # Save instead of show

print("finish")

