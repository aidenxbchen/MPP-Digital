import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 1. 读取灰度图像
image = cv2.imread('./Image//241015/1-1-4-cropped.jpg', cv2.IMREAD_GRAYSCALE)

# 2. 将图像转换为一维数据，用于 K-means 聚类
pixels = image.reshape((-1, 1))
pixels = np.float32(pixels)

# 3. 定义 K-means 聚类的参数
k = 5  # 定义要分成的区域数量
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 4. 将聚类结果转化回图像格式
segmented_image = labels.reshape(image.shape)

# 5. 在热图和彩色分割图上应用聚类结果
heatmap = np.zeros_like(image, dtype=np.float32)
color_segmented_image = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))

# 使用固定颜色映射不同区域
fixed_colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189)]  # 使用固定的颜色列表

# 对每个区域计算亮度并更新热图和显示
region_brightness = []
for label in range(k):  # 遍历每个 K-means 分割的区域
    # 创建掩膜，用于在原图中提取该区域
    mask = (segmented_image == label)

    # 计算原始灰度图像中该区域的平均亮度
    mean_brightness = np.mean(image[mask])
    region_brightness.append((label, mean_brightness))

    # 更新热图
    heatmap[mask] = mean_brightness

    # 在彩色分割图上涂上不同的颜色
    b, g, r = fixed_colors[label % len(fixed_colors)]
    color_segmented_image[mask] = (r, g, b)

# 按照亮度大小排序
region_brightness.sort(key=lambda x: x[1])

# 7. 使用 Matplotlib 绘制三个子图
plt.figure(figsize=(30, 10))

# 显示原始图像
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

# 显示基于原始图像区域平均亮度的热图
plt.subplot(1, 3, 2)
plt.imshow(heatmap, cmap='bwr', vmin=150, vmax=255)  # 使用蓝白红颜色映射，范围设定为150-255
plt.title('Heatmap of Region Brightness (Based on Original Image)')

# 显示彩色分割图
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(color_segmented_image, cv2.COLOR_BGR2RGB))
plt.title('Segmented Image with Unique Colors for Each Region')

# 添加图例，标注每种颜色的平均亮度
handles = [plt.Line2D([0], [0], color=np.array(fixed_colors[region[0]]) / 255, lw=4, label=f'Region {region[0] + 1}: {region[1]:.1f}') for region in region_brightness]
plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

plt.tight_layout()
plt.savefig('./Image//241015/1-1-4-segmentation_results.png')  # Save the image with all three visualizations
