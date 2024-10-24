import cv2
import numpy as np

image = cv2.imread('/home/aiden/MPP_digital/project/first_image.png')
gray_bgr = cv2.imread('/home/aiden/MPP_digital/project/second_image.png')

merged_image = np.hstack((image, gray_bgr))

cv2.imwrite('/home/aiden/MPP_digital/project/result_image.png', merged_image)
