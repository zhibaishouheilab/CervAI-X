"""
revised 2024/07/06
This script processes MRI images and their corresponding segmentation labels to generate K-line images 
and check for intersections with specific anatomical regions (labels 5 and 7).

Functions:
- connectedComponents_locate: Locates and sorts connected components in the image.
- adjust_angle: Adjusts angles greater than 60 degrees.
- draw_parallel_line: Draws a line parallel to the given points and extends it.
- find_intersections: Finds intersections between a line and an SC region.
- check_intersection_with_label: Checks if a line intersects with labels 5 or 7 (5 is the protusion and 7 is the depressor).
- process_files: Processes all MRI and segmentation files, draws lines, checks intersections, and saves results.

Inputs:
- MRI images and segmentation labels (in .nii.gz format).

Outputs:
- K-line images with lines drawn on them (saved as .png files).
- A text file listing names of files where intersections with labels 5 or 7 occur.
"""

"""
revised 2024/09/02
原本使用的是椎体的角点来定位上下缘的两端，但存在角点连线可能会偏
现在使用椎间盘最小矩形的顶点
为什么不用椎体，因为椎体的最小外接矩形的边也会偏的很严重
"""

import os
import numpy as np
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def xy_convent(points):
    """
    Correct the coordinates by swapping x and y if necessary.
    Args:
        points: A numpy array of shape (4, 2) representing the four corners.
    Returns:
        corrected_points: A numpy array of shape (4, 2) with corrected coordinates.
    """
    corrected_points = np.zeros_like(points)
    corrected_points[:, 0] = points[:, 1]  # Swap x and y
    corrected_points[:, 1] = points[:, 0]
    return corrected_points


def find_extreme_points(box):
    """
    根据轮廓点集，将点分为左右两部分，并分别找到左边最高点、左边最低点、右边最高点和右边最低点。
    
    Args:
    - box (numpy.ndarray): 轮廓点集，形状为 (N, 2)，其中 N 为点的数量。

    Returns:
    - left_highest (tuple): 左边最高点的坐标 (x, y)。
    - left_lowest (tuple): 左边最低点的坐标 (x, y)。
    - right_highest (tuple): 右边最高点的坐标 (x, y)。
    - right_lowest (tuple): 右边最低点的坐标 (x, y)。
    """
    # 计算V_C7的中心点
    center_x = np.mean(box[:, 0])

    # 将C7_box中的点分为左右两部分
    left_points = box[box[:, 0] < center_x]
    right_points = box[box[:, 0] >= center_x]

    # 如果left_points和right_points不为空，则找到每部分的最高点和最低点
    if left_points.size > 0:
        left_highest = tuple(left_points[np.argmin(left_points[:, 1])])  # y最小的点，即最高点
        left_lowest = tuple(left_points[np.argmax(left_points[:, 1])])   # y最大的点，即最低点
    else:
        left_highest = left_lowest = None

    if right_points.size > 0:
        right_highest = tuple(right_points[np.argmin(right_points[:, 1])])  # y最小的点，即最高点
        right_lowest = tuple(right_points[np.argmax(right_points[:, 1])])   # y最大的点，即最低点
    else:
        right_highest = right_lowest = None

    return left_highest, left_lowest, right_highest, right_lowest

def connectedComponents_locate(img, size=100):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    sorted_indices = np.argsort(centroids[1:, 1])  # 根据质心的y坐标排序
    sorted_indices = sorted_indices[stats[1:][sorted_indices][:, cv2.CC_STAT_AREA] >= size]  # 过滤掉小于size的连通域
    num_labels = len(sorted_indices)
    new_labels = np.zeros_like(labels)
    new_stats = np.zeros([num_labels, 5])
    new_centroids = np.zeros([num_labels, 2])
    new_stats[0] = stats[0]
    new_centroids[0] = centroids[0]
    for new_label, old_label in enumerate(sorted_indices):
        new_labels[labels == old_label + 1] = new_label + 1
    new_stats[:] = stats[1:][sorted_indices]
    new_centroids[:] = centroids[1:][sorted_indices]
    return num_labels, new_labels, new_stats, new_centroids

def adjust_angle(angle):
    if angle > 60:
        return angle - 90
    return angle

def draw_parallel_line(mr_middle_color, pt1, pt2, img_width, label_color):
    if pt2[0] != pt1[0]:  # avoid division by zero
        slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
        pt2 = (img_width - 1, int(pt1[1] + slope * (img_width - 1 - pt1[0])))
    else:
        slope = 0  # 默认slope为0
        pt2 = (img_width - 1, pt1[1])
    
    cv2.line(mr_middle_color, pt1, pt2, label_color, 2)
    return pt1, pt2, slope

def find_intersections(sc_image, pt1, pt2, slope):
    intersections = []
    y1, y2 = pt1[1], pt2[1]
    x1, x2 = pt1[0], pt2[0]
    for x in range(x1, x2+1):
        y = int(y1 + slope * (x - x1))
        if 0 <= y < sc_image.shape[0] and 0 <= x < sc_image.shape[1] and sc_image[y, x] > 0:
            intersections.append((x, y))
    return intersections

def check_intersection_with_label(sc_image, pt1, pt2, slope, label_image):
    y1, y2 = pt1[1], pt2[1]
    x1, x2 = pt1[0], pt2[0]
    for x in range(x1, x2+1):
        y = int(y1 + slope * (x - x1))
        if 0 <= y < label_image.shape[0] and 0 <= x < label_image.shape[1]:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if 0 <= y+dy < sc_image.shape[0] and 0 <= x+dx < sc_image.shape[1]:
                        if label_image[y+dy, x+dx] == 1:
                            #print(x,y)
                            return True
            
    return False

def process_files(mr_folder, seg_folder, k_line_folder, k_minus_file):
    mr_files = [f for f in os.listdir(mr_folder) if f.endswith('.nii.gz')]
    seg_files = [f for f in os.listdir(seg_folder) if f.endswith('.nii.gz')]
    
    intersecting_files = []
    results_list = []
    
    for seg_file in seg_files:
        base_name = seg_file.split('.nii.gz')[0]
        #if base_name!='yangguizhu_1':
        #    continue
        if not os.path.exists(os.path.join(mr_folder, base_name + '_0000.nii.gz')):
            mr_file = seg_file
        else:
            mr_file = base_name + '_0000.nii.gz'
        
        seg_img = sitk.ReadImage(os.path.join(seg_folder, seg_file))
        seg_img_np = sitk.GetArrayFromImage(seg_img)
        mr_img = sitk.ReadImage(os.path.join(mr_folder, mr_file))
        mr_img_np = sitk.GetArrayFromImage(mr_img)
        
        middle_layer = seg_img_np.shape[0] // 2
        seg_middle = seg_img_np[middle_layer, :, :]
        mr_middle = mr_img_np[middle_layer, :, :]
        
        mr_middle_normalized = ((mr_middle - np.min(mr_middle)) / (np.max(mr_middle) - np.min(mr_middle)) * 255).astype(np.uint8)
        
        if len(mr_middle_normalized.shape) == 2:
            mr_middle_color = cv2.cvtColor(mr_middle_normalized, cv2.COLOR_GRAY2BGR)
        else:
            mr_middle_color = mr_middle_normalized
        
        IVD = (seg_middle == 1).astype(np.uint8)
        SC = ((seg_middle == 3) | (seg_middle == 6) | (seg_middle == 4)).astype(np.uint8)
        labels_5_7 = ((seg_middle == 5) | (seg_middle == 7)).astype(np.uint8)
        
        # 这里写的是v，实际上是ivd
        num_V, V_labels, V_stats, V_centroids = connectedComponents_locate(IVD)
        
        if num_V > 6:
            num_V = 6
            V_centroids = V_centroids[:6]
            V_labels[V_labels > 6] = 0
            V_stats = V_stats[:7]
            V_names = ['C{}'.format(i+2) for i in range(num_V)]
        
        V_bboxes = [cv2.minAreaRect(np.argwhere(V_labels == i+1)) for i in range(num_V)]
        #IVD_bboxes = [cv2.minAreaRect(np.argwhere(IVD_labels == i+1)) for i in range(num_V)]
        #print(len(V_bboxes))
        v2_bbox = V_bboxes[0]
        v7_bbox = V_bboxes[4]
        
        v2_points = cv2.boxPoints(v2_bbox)
        v7_points = cv2.boxPoints(v7_bbox)
        v2_points = xy_convent(v2_points)
        v7_points = xy_convent(v7_points)
        
        
        v2_left_highest, v2_left_lowest, v2_right_highest, v2_right_lowest = find_extreme_points(v2_points)
        v7_left_highest, v7_left_lowest, v7_right_highest, v7_right_lowest = find_extreme_points(v7_points)
        
        V_C2 = (V_labels == 1).astype(np.uint8)
        V_C7 = (V_labels == 6).astype(np.uint8)
        
        #C2_contours, _ = cv2.findContours(V_C2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #C2_cont = C2_contours[0]
        #C2_epsilon = 0.04 * cv2.arcLength(C2_cont, True)
        #C2_box = cv2.approxPolyDP(C2_cont, C2_epsilon, True)
        #C2_box = C2_box.squeeze()
        #sorted_C2box = C2_box[np.lexsort(-C2_box.T)]
        #c2_left_highest, c2_left_lowest, c2_right_highest, c2_right_lowest = find_extreme_points(C2_box)
        
        #C7_contours, _ = cv2.findContours(V_C7, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #C7_cont = C7_contours[0]
        #C7_epsilon = 0.04 * cv2.arcLength(C7_cont, True)
        #C7_box = cv2.approxPolyDP(C7_cont, C7_epsilon, True)
        #C7_box = C7_box.squeeze()
        #c7_left_highest, c7_left_lowest, c7_right_highest, c7_right_lowest = find_extreme_points(C7_box)
        
        c2_bottom_left = v2_left_highest
        c2_bottom_right = v2_right_highest
        c7_top_left = v7_left_lowest
        c7_top_right = v7_right_lowest
        
        c2_pt1, c2_pt2, c2_slope = draw_parallel_line(mr_middle_color, (int(c2_bottom_left[0]), int(c2_bottom_left[1])), (int(c2_bottom_right[0]), int(c2_bottom_right[1])), mr_middle_color.shape[1], (0, 0, 255))
        c7_pt1, c7_pt2, c7_slope = draw_parallel_line(mr_middle_color, (int(c7_top_left[0]), int(c7_top_left[1])), (int(c7_top_right[0]), int(c7_top_right[1])), mr_middle_color.shape[1], (0, 0, 255))
        
        c2_intersections = find_intersections(SC, c2_pt1, c2_pt2, c2_slope)
        c7_intersections = find_intersections(SC, c7_pt1, c7_pt2, c7_slope)
        
        k_minus = 0
        if len(c2_intersections) > 0 and len(c7_intersections) > 0:
            c2_mid = c2_intersections[len(c2_intersections)//2]
            c7_mid = c7_intersections[len(c7_intersections)//2]
            k_slope = (c7_mid[1] - c2_mid[1]) / (c7_mid[0] - c2_mid[0]) if (c7_mid[0] - c2_mid[0]) != 0 else 0
            cv2.line(mr_middle_color, c2_mid, c7_mid, (0, 255, 255), 2)  # K-line in yellow
            
            for (x, y) in c2_intersections:
                cv2.circle(mr_middle_color, (x, y), 3, (255, 255, 255), -1)
            for (x, y) in c7_intersections:
                cv2.circle(mr_middle_color, (x, y), 3, (255, 255, 255), -1)
            
            if check_intersection_with_label(seg_middle, c2_mid, c7_mid, k_slope, labels_5_7):
                print(f'{base_name} has K-.******************************')
                intersecting_files.append(base_name)
                k_minus = 1
            else:
                print(f'{base_name} has no K-')

        results_list.append({'Name': base_name, 'K_minus': k_minus})
   
        plt.imshow(mr_middle_color, cmap='gray')
        plt.title(f'{base_name} K-line')
        plt.axis('off')
        plt.savefig(os.path.join(k_line_folder, f'{base_name}_k_line.png'))
    
    with open(k_minus_file, 'w') as f:
        for item in intersecting_files:
            f.write("%s\n" % item)
            
    # Convert results to DataFrame and save to Excel
    df_results = pd.DataFrame(results_list)
    df_results.to_excel(k_minus_file.replace('txt','xlsx'), index=False)
    print(f"Results saved to {k_minus_file.replace('txt','xlsx')}")

def main():
    parser = argparse.ArgumentParser(description='处理模型预测结果并生成诊断图')
    parser.add_argument('--main_folder', type=str, help='包含模型文件夹的根目录路径')
    parser.add_argument('--save_folder', type=str, help='保存结果的根目录路径')
    args = parser.parse_args()
    
    mr_dir = os.path.join(args.main_folder, 'img')
    seg_dir = os.path.join(args.main_folder, 'pred_protrude')
    save_dir = args.save_folder
    os.makedirs(save_dir, exist_ok=True)
    
    process_files(mr_dir, seg_dir, save_dir, os.path.join(save_dir, 'k_minus.txt'))
    
    
if __name__ == '__main__':
    main()
    