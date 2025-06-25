import os
import numpy as np
from scipy.ndimage import label, center_of_mass

def generate_heatmap(segmentation_map, target_label=5, sigma_scale=0.1, min_region_size=10):
    """
    根据分割图生成热力图，其中每个连通域的sigma与其大小相关。
    :param segmentation_map: 输入分割图，2D numpy数组
    :param target_label: 标签为5的区域为异常区域
    :param sigma_scale: sigma与连通域大小的比例系数
    :return: 返回生成的总热力图
    """
    # 提取标签为5的区域
    binary_map = segmentation_map == target_label
    
    # 连通域检测
    labeled_map, num_features = label(binary_map)
    
    # 初始化热力图
    heatmap = np.zeros_like(segmentation_map, dtype=np.float32)
    
    # 遍历每个连通域
    for region in range(1, num_features + 1):
        # 获取当前连通域的掩码
        region_mask = labeled_map == region
        
        # 计算连通域的中心点
        y_mu, x_mu = center_of_mass(region_mask)
        
        # 计算连通域的大小（像素数）
        region_size = np.sum(region_mask)
        
        # 过滤掉小于最小连通域大小的区域
        if region_size < min_region_size:
            continue
        
        # 根据连通域的大小计算sigma
        sigma = sigma_scale * np.sqrt(region_size)
        
        # 生成一个与输入大小相同的网格坐标
        y, x = np.indices(segmentation_map.shape)
        
        # 计算高斯热力图
        gaussian_heatmap = np.exp(-((x - x_mu) ** 2 + (y - y_mu) ** 2) / (2 * sigma ** 2))
        
        # 将高斯热力图加到总热力图中
        heatmap += gaussian_heatmap
    
    return heatmap

def process_file(file_path):
    # 读取文件
    if file_path.endswith('.npy'):
        data = np.load(file_path)
    elif file_path.endswith('.npz'):
        data = np.load(file_path)['data']
    else:
        return

    # 假设数组形状为 (2, 3, H, W)
    data_array = data[0]  # data
    target_array = data[1]  # target

    # 遍历每个切片生成热力图并插入
    heatmaps = []
    for i in range(target_array.shape[0]):
        heatmap = generate_heatmap(target_array[i],target_label=5, sigma_scale=1, min_region_size=10)
        heatmaps.append(heatmap)
    
    heatmaps = np.array(heatmaps)  # (3, H, W)

    # 在 data 和 target 之间插入热力图
    new_data = np.insert(data_array, 1, heatmaps, axis=0)

    # 将 target 中的 label=5 转换为 label=1
    target_array[target_array == 5] = 1

    # 组合新的数据集
    new_combined_array = np.stack((data_array,heatmaps, target_array), axis=0)

    # 保存文件
    if file_path.endswith('.npy'):
        np.save(file_path, new_combined_array)
    elif file_path.endswith('.npz'):
        np.savez(file_path, data=new_combined_array)

def process_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.npy') or file.endswith('.npz'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                process_file(file_path)

# 示例用法
folder_path = '/home/ubuntu/Project/nnUNet/dataset/nnUNet_preprocessed/Task902_spine/nnUNetData_plans_v2.1_2D_stage0'  # 替换为包含npy和npz文件的文件夹路径
process_folder(folder_path)
