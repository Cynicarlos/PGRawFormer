import rawpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def calculate_noise_level(clean_image, noisy_image):
    #return np.abs(clean_image - noisy_image)
    return clean_image - noisy_image

def calculate_statistics(clean_image, noisy_image, block_size, overlap):
    brightness_values = []
    noise_levels = []
    
    height, width = clean_image.shape
    step_x = block_size[1] - overlap
    step_y = block_size[0] - overlap
    
    for i in range(0, height - block_size[0] + 1, step_y):
        for j in range(0, width - block_size[1] + 1, step_x):
            x_start = j
            y_start = i
            region_clean = clean_image[y_start:y_start + block_size[0], x_start:x_start + block_size[1]]
            region_noisy = noisy_image[y_start:y_start + block_size[0], x_start:x_start + block_size[1]]
            brightness_values.append(np.mean(region_clean))
            noise_levels.append(np.mean(calculate_noise_level(region_clean, region_noisy)))
    
    return brightness_values, noise_levels

input_path = 'E:\Deep Learning\datasets\RAW\SID\Sony\Sony\short/00026_00_0.1s.ARW'
gt_path = 'E:\Deep Learning\datasets\RAW\SID\Sony\Sony\long/00026_00_10s.ARW'
input_raw = rawpy.imread(input_path).raw_image_visible.astype(np.uint16)
gt_raw = rawpy.imread(gt_path).raw_image_visible.astype(np.uint16)

input_raw = (np.float32(input_raw) - 512) / np.float32(16383 - 512)
gt_raw = (np.float32(gt_raw) - 512) / np.float32(16383 - 512)

input_raw = input_raw * 100
#input_raw = input_raw
input_raw = np.maximum(np.minimum(input_raw, 1.0), 0.0)
gt_raw = np.maximum(np.minimum(gt_raw, 1.0), 0.0)

# 计算亮度值和噪声水平
brightness_values, noise_levels = calculate_statistics(gt_raw, input_raw, [16, 16], 8)

# 计算相关性
correlation, _ = pearsonr(brightness_values, noise_levels)
print(f"Correlation coefficient: {correlation}")

# 显示相关性图
plt.figure(figsize=(8, 6))
plt.scatter(brightness_values, noise_levels, color='blue')
plt.title('Brightness vs Diff Level')
plt.xlabel('Brightness')
plt.ylabel('Diff Level')
plt.grid(True)
plt.show()

# 过滤出大于阈值的噪声水平
filtered_noise_levels = [level for level in noise_levels if level >= 0.04 or level <= -0.04]

# 绘制过滤后的噪声水平分布直方图
plt.figure(figsize=(8, 6))
plt.hist(filtered_noise_levels, bins=50, color='blue', alpha=0.7)
plt.title(f'Distribution of Diff Level >= 0.04 or <= -0.04')
plt.xlabel('Diff Level')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#input_raw = np.ascontiguousarray(input_raw)
#gt_raw = np.ascontiguousarray(gt_raw)

# 计算差异
# difference = np.abs(input_raw - gt_raw)

# # 显示差异图像
# plt.imshow(difference, cmap='gray')
# plt.title('Difference Image')
# plt.show()

# # 计算并打印统计数据
# mean_diff = np.mean(difference)
# max_diff = np.max(difference)
# print(f"Mean difference: {mean_diff}")
# print(f"Max difference: {max_diff}")

# # 生成热图
# plt.imshow(difference, cmap='hot')
# plt.title('Heatmap of Differences')
# plt.show()