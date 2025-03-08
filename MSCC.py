"""
revised 2024/07/06
This script processes MRI images and their corresponding segmentation labels to generate middle layer images,
calculate the maximum MSCC (maximum spinal cord compression) for each segment, and save the results in an Excel file.

Functions:
- connectedComponents_locate: Locates and sorts connected components in the image.
- adjust_angle: Adjusts angles greater than 60 degrees.
- draw_parallel_line: Draws a line parallel to the given points and extends it.
- find_intersections: Finds intersections between a line and an SC region.
- distance: Calculates the Euclidean distance between two points.
- process_files: Processes all MRI and segmentation files, draws lines, checks intersections, and saves results.

Inputs:
- MRI images and segmentation labels (in .nii.gz format).

Outputs:
- Middle layer images with lines drawn on them (saved as .png files).
- An Excel file listing the maximum MSCC and corresponding segment for each file.
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
    if angle < -60:
        return angle + 90
    return angle

def draw_parallel_line(mr_middle_color, points, img_width, label_color):
    
    #我发现cv2.boxPoints()得到的点xy坐标是反过来的，所以这里用了xy_convent()函数来得到正确是xy坐标
    points = xy_convent(points)
    center = np.mean(points,axis=0)
    #print(center)
    pt1 = (int(center[0]), int(center[1]))  # 注意交换坐标轴

    left_highest, left_lowest, right_highest, right_lowest = find_extreme_points(points)
    
    #这里我们直接使用上边缘连线的slope
    delta_x = right_highest[0]-left_highest[0]
    delta_y = right_highest[1]-left_highest[1]
    if delta_x!=0:
        slope = delta_y/delta_x
        pt2 = (img_width - 1,int(pt1[1] + slope * (img_width - 1 - pt1[0])))
    else:
        slope = 0
        pt2 = (img_width - 1, pt1[0])
    
    cv2.line(mr_middle_color, pt1, pt2, label_color, 2)
    #print(pt1,pt2)
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

def distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def process_files(mr_folder, seg_folder, output_folder, output_excel_file):
    mr_files = [f for f in os.listdir(mr_folder) if f.endswith('.nii.gz')]
    seg_files = [f for f in os.listdir(seg_folder) if f.endswith('.nii.gz')]
    
    results = []
    
    for seg_file in seg_files:
        base_name = seg_file.split('.nii.gz')[0]
        #if base_name!='liaolujian_2':
        #    continue
        if not os.path.exists(os.path.join(mr_folder, base_name + '_0000.nii.gz')):
            mr_file = seg_file
        else:
            mr_file = base_name + '_0000.nii.gz'
        
        if mr_file not in mr_files:
            continue
        
        seg_img = sitk.ReadImage(os.path.join(seg_folder, seg_file))
        seg_img_np = sitk.GetArrayFromImage(seg_img)
        mr_img = sitk.ReadImage(os.path.join(mr_folder, mr_file))
        mr_img_np = sitk.GetArrayFromImage(mr_img)
        
        # Assume the middle layer is at seg_img_np.shape[0] // 2
        middle_layer = seg_img_np.shape[0] // 2
        seg_middle = seg_img_np[middle_layer, :, :]
        mr_middle = mr_img_np[middle_layer, :, :]
        
        # Normalize the MR image slice
        mr_middle_normalized = ((mr_middle - np.min(mr_middle)) / (np.max(mr_middle) - np.min(mr_middle)) * 255).astype(np.uint8)
        
        # Ensure the image is in 3-channel format for drawing lines
        if len(mr_middle_normalized.shape) == 2:
            mr_middle_color = cv2.cvtColor(mr_middle_normalized, cv2.COLOR_GRAY2BGR)
        else:
            mr_middle_color = mr_middle_normalized
        
        V = (seg_middle == 2).astype(np.uint8)  # Assume label 2 for V
        IVD = (seg_middle == 1).astype(np.uint8)  # Assume label 1 for IVD
        SC = ((seg_middle == 3) | (seg_middle == 6)).astype(np.uint8)  # Assume label 3 and 6 for SC
        
        num_V, V_labels, V_stats, V_centroids = connectedComponents_locate(V)
        num_IVD, IVD_labels, IVD_stats, IVD_centroids = connectedComponents_locate(IVD)
        
        # Limit to C2-C7 and C2-C3 to C6-C7
        if num_V > 6:
            num_V = 6
            V_centroids = V_centroids[:6]
            V_labels[V_labels > 6] = 0
            V_stats = V_stats[:7]
            V_names = ['C{}'.format(i+2) for i in range(num_V)]
        if num_IVD > 5:
            num_IVD = 5
            IVD_centroids = IVD_centroids[:5]
            IVD_labels[IVD_labels > 5] = 0
            IVD_stats = IVD_stats[:6]
            IVD_names = ['C{}-C{}'.format(i+2, i+3) for i in range(num_IVD)]
        
        # Check if IVD is between corresponding V
        for i in range(num_IVD):
            ivd_y = IVD_centroids[i][1]
            v1_y = V_centroids[i][1]
            v2_y = V_centroids[i+1][1]
            if not (v1_y < ivd_y < v2_y):
                print(f'{base_name} segment {IVD_names[i]} not between {V_names[i]} and {V_names[i+1]}')
        
        # Extracting bounding boxes and centers
        V_bboxes = [cv2.minAreaRect(np.argwhere(V_labels == i+1)) for i in range(num_V)]
        IVD_bboxes = [cv2.minAreaRect(np.argwhere(IVD_labels == i+1)) for i in range(num_IVD)]
        
        # Drawing lines and calculating lengths
        mscc_values = []
        img_width = mr_middle_color.shape[1]
        for i in range(num_IVD):
            ivd_bbox = IVD_bboxes[i]
            v1_bbox = V_bboxes[i]
            v2_bbox = V_bboxes[i+1]

            ivd_points = cv2.boxPoints(ivd_bbox)
            #print(ivd_points)
            ivd_pt1, ivd_pt2, ivd_slope = draw_parallel_line(mr_middle_color, ivd_points, img_width, (0, 0, 255))
            v1_points = cv2.boxPoints(v1_bbox)
            v1_pt1, v1_pt2, v1_slope = draw_parallel_line(mr_middle_color, v1_points, img_width, (0, 255, 0))
            v2_points = cv2.boxPoints(v2_bbox)
            v2_pt1, v2_pt2, v2_slope = draw_parallel_line(mr_middle_color, v2_points, img_width, (0, 255, 0))

            # Ensure SC is not empty
            if SC.max() == 0:
                mscc_values.append(-1)
                continue

            # Find the intersection of the SC line with the SC region
            ivd_intersections = find_intersections(SC, ivd_pt1, ivd_pt2, ivd_slope)
            v1_intersections = find_intersections(SC, v1_pt1, v1_pt2, v1_slope)
            v2_intersections = find_intersections(SC, v2_pt1, v2_pt2, v2_slope)
            
            if len(ivd_intersections) == 0 or len(v1_intersections) == 0 or len(v2_intersections) == 0:
                mscc_values.append(-1)
                continue
            
            ivd_1, ivd_2 = ivd_intersections[0], ivd_intersections[-1]
            v1_1, v1_2 = v1_intersections[0], v1_intersections[-1]
            v2_1, v2_2 = v2_intersections[0], v2_intersections[-1]
            
            length_ivd = distance(ivd_1, ivd_2)
            length_v1 = distance(v1_1, v1_2)
            length_v2 = distance(v2_1, v2_2)
            
            # Draw the lines on the image
            cv2.line(mr_middle_color, (ivd_1[0], ivd_1[1]), (ivd_2[0], ivd_2[1]), (255, 0, 0), 2)
            cv2.line(mr_middle_color, (v1_1[0], v1_1[1]), (v1_2[0], v1_2[1]), (255, 0, 0), 2)
            cv2.line(mr_middle_color, (v2_1[0], v2_1[1]), (v2_2[0], v2_2[1]), (255, 0, 0), 2)
            
            # Draw intersection points with red circles
            for (x, y) in ivd_intersections:
                cv2.circle(mr_middle_color, (x, y), 3, (255, 255, 255), -1)
            for (x, y) in v1_intersections:
                cv2.circle(mr_middle_color, (x, y), 3, (255, 255, 255), -1)
            for (x, y) in v2_intersections:
                cv2.circle(mr_middle_color, (x, y), 3, (255, 255, 255), -1)
            
            if length_v1 == 0 or length_v2 == 0:
                mscc_values.append(-1)
            else:
                mscc = (1 - 2 * length_ivd / (length_v1 + length_v2)) * 100
                mscc_values.append(mscc)
            
            # Print lengths
            print(f'{IVD_names[i]} IVD length: {length_ivd}')
            print(f'{V_names[i]} V1 length: {length_v1}')
            print(f'{V_names[i+1]} V2 length: {length_v2}')
        
        # Draw rectangles
        for bbox in V_bboxes + IVD_bboxes:
            box_points = cv2.boxPoints(bbox)
            box_points = np.int0(box_points)
            box_points[:, [0, 1]] = box_points[:, [1, 0]]  # 交换坐标轴
            cv2.drawContours(mr_middle_color, [box_points], 0, (255, 255, 0), 2)
        
        # Save output image
        plt.imshow(mr_middle_color, cmap='gray')
        plt.title(f'{base_name} Middle Layer')
        plt.axis('off')
        plt.savefig(os.path.join(output_folder, f'{base_name}_middle_layer.png'))
        
        # Find the maximum MSCC and corresponding segment
        if mscc_values:
            max_mscc = max(mscc_values)
            max_index = mscc_values.index(max_mscc)
            max_segment = IVD_names[max_index]
            results.append((base_name, max_mscc, max_segment))
        
        # Output MSCC values
        for name, mscc in zip(IVD_names, mscc_values):
            print(f'{name} MSCC: {mscc}')
    
    # Sort results by file name
    results.sort(key=lambda x: x[0])
    
    # Save results to Excel file
    df = pd.DataFrame(results, columns=['FileName', 'MaxMSCC', 'Segment'])
    df.to_excel(output_excel_file, index=False)
    
    
def main():
    parser = argparse.ArgumentParser(description='处理模型预测结果并生成诊断图')
    parser.add_argument('main_folder', type=str, help='包含模型文件夹的根目录路径')
    parser.add_argument('save_folder', type=str, help='保存结果的根目录路径')
    args = parser.parse_args()
    
    mr_dir = os.path.join(args.main_folder, 'img')
    seg_dir = os.path.join(args.main_folder, 'pred_protrude')
    output_folder = os.path.join(args.save_folder, 'mscc')
    output_excel_file = os.path.join(output_folder, 'mscc_results.xlsx')
    os.makedirs(output_folder, exist_ok=True)
    
    process_files(mr_dir, seg_dir, output_folder, output_excel_file)
    


if __name__ == '__main__':
    main()