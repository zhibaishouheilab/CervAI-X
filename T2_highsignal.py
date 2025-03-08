"""
This script performs protrusion segmentation and signal intensity analysis on MRI images of the cervical spine.
It extracts the segmentation of IVD protrusions, calculates the signal intensity along the spinal cord, and generates plots.

Functionality:
- Segment the IVD protrusions
- Calculate signal intensity and T2-MI, RSCI values
- Generate and save plots with annotations

Usage:
- Define the input and output paths.
- Run the script to process all the MRI images in the input path.
"""

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import os
import cv2
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist
import json
import pandas as pd
import argparse

def get_signal(image, label, SC_label, HS_label):
    SC_exist = True
    if np.sum(label == SC_label) < 100:
        SC_exist = False
        return 0, 0, 0, SC_exist
        
    ymin = min(np.where(label == SC_label)[0])
    ymax = max(np.where(label == SC_label)[0])
    y_axis = np.arange(ymin + 2, ymax - 1)
    sig_array = []
    p_cood = np.argwhere((label == SC_label) + (label == HS_label))
    sig_sum = 0
    
    for i in range(p_cood.shape[0]):
        sig_sum += image[p_cood[i][0], p_cood[i][1]]
        
    sig_mean = sig_sum / p_cood.shape[0]
    
    for i in range(ymin + 2, ymax - 1, 1):
        signal_sum = 0
        num_point = 0
        for m in range(5):
            if np.sum((label[i + m - 2] == SC_label) | (label[i + m - 2] == HS_label)) == 0:
                num_point += 1
            else:      
                xmin = min(np.where((label[i + m - 2] == SC_label) | (label[i + m - 2] == HS_label))[0])
                xmax = max(np.where((label[i + m - 2] == SC_label) | (label[i + m - 2] == HS_label))[0])
                num_point += xmax - xmin + 1

                for j in range(xmin, xmax + 1, 1):
                    signal_sum += image[i, j]
        
        signal_avg = signal_sum / num_point
        if signal_avg < np.mean(sig_array) / 2:
            if len(sig_array) > 0:
                signal_avg = np.mean(sig_array[-3:-1])
        
        sig_array.append(signal_avg)
    
    return y_axis, sig_array, sig_mean, SC_exist

def signal_range(label, HS_label):
    if HS_label in label:
        ymin = min(np.where(label == HS_label)[0])
        ymax = max(np.where(label == HS_label)[0])
    else:
        ymin = 0
        ymax = 1
    
    return ymin, ymax

def remove_element(arr, m):
    return np.append(arr[:m], arr[m + 1:])

def get_centerIVD(label_array, IVD_label, size):
    IVD = np.zeros(label_array.shape, np.uint8)
    IVD[label_array == IVD_label] = 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(IVD)
    for m in range(num_labels):
        if stats[:, -1][m] < size:
            IVD[labels == m] = 0
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(IVD)
    sorted_center = sorted(centroids[1:, 1])
    return num_labels - 1, sorted_center

def get_center(label_array, IVD_label):
    IVD = np.zeros(label_array.shape, np.uint8)
    IVD[label_array == IVD_label] = 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(IVD)
    size = 10
    for m in range(num_labels):
        if stats[:, -1][m] < size:
            IVD[labels == m] = 0
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(IVD)
    sorted_indices = np.argsort(centroids[1:, 1])
    sorted_centroids = centroids[1:][sorted_indices]
    return num_labels - 1, sorted_centroids

def spine_curve(center_points, ymin, ymax):
    x = center_points[:, 0]
    y = center_points[:, 1]
    cs = CubicSpline(y, x)
    interp_y = np.linspace(ymin, ymax, 1000)
    interp_x = cs(interp_y)
    return np.column_stack((interp_x, interp_y))

def curve_signal(image, label, SC_label, HS_label, curve):
    SC_exist = True
    if np.sum(label == SC_label) < 100:
        SC_exist = False
        return 0, 0, 0, SC_exist
    curve_points = np.array(curve)
    label_points = np.argwhere((label == SC_label) | (label == HS_label))
    distances = cdist(curve_points, label_points)
    nearest_label_indices = np.argmin(distances, axis=1)

    average_pixel_values = []
    for i in range(len(curve_points)):
        nearest_label_indices_for_point = np.where(nearest_label_indices == i)[0]
        nearest_label_points = label_points[nearest_label_indices_for_point]
        pixel_values = image[nearest_label_points[:, 0], nearest_label_points[:, 1]]
        average_pixel_value = np.mean(pixel_values)
        average_pixel_values.append(average_pixel_value)
    
    y_axis = np.arange(0, len(average_pixel_values))
    return y_axis, average_pixel_values, np.mean(average_pixel_values), SC_exist

def calculate_t2_mi(label_array, V_label, y_axis, sig_array):
    num_V, V_center = get_centerIVD(label_array, V_label, 20)
    #print(num_V,V_center)
    V_C = []
    t2_mi_list = []
    y_min = np.min(y_axis)
    y_max = np.max(y_axis)
    #print(y_min,y_max)
    for i in range(num_V - 1):
        ymin = V_center[i]
        ymax = V_center[i + 1]
        #print(ymin,ymax)
        if ymin < y_min:
            ymin = y_min
        if ymax > y_max:
            ymax = y_max
        if ymin >= y_max or ymax <= y_min:
            continue
        #print(ymax-y_min)
        #print(np.floor(ymin - y_min),np.ceil(ymax - y_min))
        t2_si_values = sig_array[int(np.floor(ymin - y_min)):int(np.ceil(ymax - y_min))]
        #print(ymin,ymax)
        t2_si_range = np.max(t2_si_values) - np.min(t2_si_values)
        average_absolute_t2_si = np.mean(np.abs(t2_si_values))
        t2_mi = (t2_si_range * 100) / average_absolute_t2_si
        t2_mi_list.append(round(t2_mi, 1))
        V_C.append(ymin)
        V_C.append(ymax)
        
    N_V = len(t2_mi_list)
    if N_V > 1:
        t2_mi_right = t2_mi_list[1]
        t2_mi_relative = t2_mi_list[0] * 2 / (t2_mi_right + t2_mi_list[0])
        t2_mi_relative_list = [round(t2_mi_relative, 1)]
    else:
        t2_mi_relative_list = [1]
    
    for i in range(1, N_V - 1):
        t2_mi_left = t2_mi_list[i - 1]
        t2_mi_right = t2_mi_list[i + 1]
        t2_mi_relative = t2_mi_list[i] * 3 / (t2_mi_left + t2_mi_right + t2_mi_list[i])
        t2_mi_relative_list.append(round(t2_mi_relative, 1))
    
    if N_V > 1:
        t2_mi_left = t2_mi_list[N_V - 2]
        t2_mi_relative = t2_mi_list[N_V - 1] * 2 / (t2_mi_left + t2_mi_list[N_V - 1])
        t2_mi_relative_list.append(round(t2_mi_relative, 1))
    
    return N_V, V_C, t2_mi_list, t2_mi_relative_list

def plot_highsignal(name, image_path, label_path, SC_label, HS_label, IVD_label, HD_label, save_path=None):
    img = sitk.ReadImage(image_path)
    img_array = sitk.GetArrayFromImage(img)
    if np.mean(img_array)<1:
        img_array*=1000
    label = sitk.ReadImage(label_path)
    label_array = sitk.GetArrayFromImage(label)
    ymin = min(np.where(label_array != 0)[1])
    ymax = max(np.where(label_array != 0)[1])
    h, x, y = img_array.shape
    all_t2_list = []
    all_rsci_list = []
    all_t2_rsci_list = []
    plt.figure(1)
    fig, ax = plt.subplots(h, 1)
    
    for i in range(h):
        #print(i)
        V_label = 2
        y_axis, sign_inten, sig_mean, SC_exist = get_signal(img_array[i], label_array[i], SC_label, HS_label)
        if not SC_exist:
            continue
        num_IVD, IVD_center = get_centerIVD(label_array[i], IVD_label, 20)
        num_HD, HD_center = get_centerIVD(label_array[i], HD_label, 2)
        num_V, V_center, t2_mi_list, t2_mi_relative_list = calculate_t2_mi(label_array[i], V_label, y_axis, sign_inten)
        t2_mi_rsci_list = [round(a * b, 1) for a, b in zip(t2_mi_list, t2_mi_relative_list)]
        
        all_t2_list.append(t2_mi_list)
        all_rsci_list.append(t2_mi_relative_list)
        all_t2_rsci_list.append(t2_mi_rsci_list)

        ax[i].text(ymin + 20, min(sign_inten) - 20, 'T2-MI', color='g', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        ax[i].text(ymin + 20, min(sign_inten) - 50, 'RSCI', color='b', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        ax[i].text(ymin + 20, min(sign_inten) - 80, 'T2*RSCI', color='c', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        
        vertebrae_labels = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'T1', 'T2', 'T3', 'T4', 'T5','T6','T7','T8','T9','T10','T11']
        for m in range(num_V):
            ax[i].axvline(x=V_center[m * 2], color='g', linestyle='-.')
            ax[i].axvline(x=V_center[m * 2 - 1], color='g', linestyle='-.')
            ax[i].text((V_center[m * 2] + V_center[m * 2 + 1]) / 2, min(sign_inten) - 20, t2_mi_list[m], color='g', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
            ax[i].text((V_center[m * 2] + V_center[m * 2 + 1]) / 2, min(sign_inten) - 50, t2_mi_relative_list[m], color='b', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
            ax[i].text((V_center[m * 2] + V_center[m * 2 + 1]) / 2, min(sign_inten) - 80, t2_mi_rsci_list[m], color='c', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
            ax[i].text(V_center[m * 2], max(sign_inten) + 30, vertebrae_labels[m], color='g', ha='center', va='bottom', fontsize=10, fontweight='bold')
            if m < 5 and t2_mi_list[m] >= 22.5 and t2_mi_relative_list[m] >= 1.2:
                ax[i].text((V_center[m * 2] + V_center[m * 2 + 1]) / 2, min(sign_inten) + 25, 'pred high', color='r', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
        HS_min, HS_max = signal_range(label_array[i], HS_label)

        if HS_min != 0:
            if HS_max - HS_min < 15:
                HS_max += 7
                HS_min -= 7
            ax[i].text((HS_min + HS_max) / 2, max(sign_inten) + 30, 'T2 high', color='r', ha='center', va='top', fontsize=12, fontweight='bold')
            
        plt.sca(ax[i])
        plt.ylim(min(sign_inten) - 70, max(sign_inten) + 70)
        plt.xlim(ymin, ymax)
        plt.axhline(y=sig_mean, linestyle='--', color='purple', alpha=0.2)
        plt.plot(y_axis, sign_inten, 'b--')
        
        if i == 0:
            plt.text(ymin, max(sign_inten) + 90, 'T2 signal intensity', ha='center', va='center', fontweight='bold')
        if i < h - 1:
            plt.xticks([])
        else:
            plt.xlabel('y-axis coordinates', fontweight='bold')
        
        ax[i].text(ymin - 30, (min(sign_inten) + max(sign_inten)) / 2, f'Layer {i + 1}', color='black', ha='right', va='center', rotation=90)

    plt.subplots_adjust(hspace=0.1)
    plt.show()
    
    if save_path is None:
        fig.savefig('%s.jpg' % name, dpi=300)
    else:
        fig.savefig(os.path.join(save_path, '%s.jpg' % name), dpi=300)
    
    plt.close(fig)
    return all_t2_list, all_rsci_list, all_t2_rsci_list

def exist_highsignal(label_path, HS_label):
    label = sitk.ReadImage(label_path)
    label_array = sitk.GetArrayFromImage(label)
    
    if HS_label in label_array:
        HS_loc = np.where(label_array == HS_label)
        print("high signal exists in %s slice" % np.unique(HS_loc[0]))
        return True
    else:
        print("No high signal")
        return False


def plot_folder(folder_a_path, folder_b_path, save_path):
    files_a = [file for file in os.listdir(folder_a_path) if file.endswith('.nii.gz')]
    
    result_dict = {}  # 初始化一个空的字典用于存储结果
    result_list = []  # 用于存储每行数据的列表，最后保存到Excel

    vertebrae_columns = ['C2-3', 'C3-4', 'C4-5', 'C5-6', 'C6-7', 'C7-T1']  # 定义列名

    for file_a in files_a:
        name = file_a.replace('.nii.gz', "")
        # 如果已经存在保存的文件，则跳过处理
        print("Now processing:", file_a.replace('.nii.gz', ""))
        file_a_path = os.path.join(folder_a_path, file_a)
        
        if not os.path.exists(os.path.join(folder_b_path, file_a.replace('.nii.gz', '_0000.nii.gz'))):
            file_b = file_a
        else:
            file_b = file_a.replace('.nii.gz', '_0000.nii.gz')
        
        file_b_path = os.path.join(folder_b_path, file_b)
        
        # 检查高信号是否存在
        if_exsit = exist_highsignal(file_a_path, 6)
        
        # 获取T2-MI、RSCI、T2-MI*RSCI的列表
        t2_mi_list, t2_mi_relative_list, t2_mi_rsci_list = plot_highsignal(name, file_b_path, file_a_path, 3, 6, 1, 5, save_path)
        
        # 遍历每一层，填充列表，每层为一行数据
        for i, t2_mi in enumerate(t2_mi_list):
            layer_name = f"{name}_Layer{i+1}"  # 构造每层的名字

            # 创建一行的数据，其中第一列是case名加层名，后面6列是t2值，如果没有值则填0
            row_data = [layer_name] + [t2_mi[j] if j < len(t2_mi) else 0 for j in range(6)]
            result_list.append(row_data)
    
    # 创建 DataFrame
    df = pd.DataFrame(result_list, columns=['Case Name'] + vertebrae_columns)

    # 将 DataFrame 保存为Excel
    excel_save_path = os.path.join(save_path, 'result_data.xlsx')
    df.to_excel(excel_save_path, index=False)
    
    print(f"Result saved to {excel_save_path}")
    return df  # 返回DataFrame以供进一步使用

print("Data written to output_with_labels.txt.")


def main():
    parser = argparse.ArgumentParser(description='处理模型预测结果并生成诊断图')
    parser.add_argument('main_folder', type=str, help='包含模型文件夹的根目录路径')
    parser.add_argument('save_folder', type=str, help='保存结果的根目录路径')
    args = parser.parse_args()
    
    model_folder_path = args.main_folder
    img_path = os.path.join(model_folder_path, 'img')
    label_path = os.path.join(model_folder_path, 'pred_protrude')
    save_path = os.path.join(args.save_folder)
        
    os.makedirs(save_path, exist_ok=True)
    plot_folder(label_path, img_path, save_path)  # 确保已定义或导入plot_folder函数

if __name__ == '__main__':
    main()