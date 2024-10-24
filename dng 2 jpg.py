import rawpy
import cv2

# Load the DNG image using rawpy
dng_file = './Image/241012-test/3.dng'
raw = rawpy.imread(dng_file)  # Load image without any size reduction

# Postprocess the image using the half_size option to reduce size
rgb_image = raw.postprocess(use_camera_wb=True, half_size=True)

# Convert the RGB image to BGR format for OpenCV
bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

# Write the image as a .jpg file, preserving details with high quality
output_file = './Image/241012-test/3.jpg'
cv2.imwrite(output_file, bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

print(f'Image saved as {output_file}')
