
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取灰度图像
image = cv2.imread('./Image/241012-test/cropped_image.png', cv2.IMREAD_GRAYSCALE)

print(np.mean(image))