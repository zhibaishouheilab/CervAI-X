"""
revised 2024/07/06
This script processes MRI images and labels to calculate Cobb angles for cervical vertebrae (C2-C7).
It reads images and labels from specified directories, computes the Cobb angles, and saves the results as images and a summary text file.

Functions:
- find_files: Find files with a specific extension in a directory.
- calculate_cobb_angle: Calculate Cobb angle given four points.
- plot_cobb_angle: Plot Cobb angle on an image.
- process_patient_C27: Process images and labels for a patient and compute Cobb angles.

Usage:
- Define the input image and label directories, and the output directory.
- Run the script to process all images and labels in the input directories and save the results.

"""

import os
import SimpleITK as sitk
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import argparse

def find_files(directory, extension):
    """
    Find files with a specific extension in a directory.

    Args:
        directory (str): Directory to search for files.
        extension (str): File extension.

    Returns:
        list: List of full paths of all matching files.
    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))
    return file_list

def calculate_cobb_angle(point1, point2, point3, point4):
    """
    Calculate Cobb angle.

    Args:
        point1, point2, point3, point4 (tuple): Coordinates of four points.

    Returns:
        float: Cobb angle.
    """
    line1 = np.array(point2) - np.array(point1)
    line2 = np.array(point4) - np.array(point3)
    angle1 = np.arctan2(line1[1], line1[0])
    angle2 = np.arctan2(line2[1], line2[0])
    cobb_angle = np.abs(np.degrees(angle2 - angle1))
    if cobb_angle > 180:
        cobb_angle = 360 - cobb_angle
    if cobb_angle > 90:
        cobb_angle = 180 - cobb_angle
    return cobb_angle

def plot_cobb_angle(point1, point2, point3, point4, fig, show_plot):
    """
    Plot Cobb angle.

    Args:
        point1, point2, point3, point4 (tuple): Coordinates of four points.
        fig: Matplotlib figure object.
        show_plot (bool): Whether to show the plot.

    Returns:
        fig: Matplotlib figure object.
    """
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r')
    plt.plot([point3[0], point4[0]], [point3[1], point4[1]], 'r')
    if show_plot:
        plt.show()
    return fig

def process_patient_C27(img_dir, label_dir, save_dir, if_show):
    """
    Process images and labels for a patient and compute Cobb angles.

    Args:
        img_dir (str): Directory of image data.
        label_dir (str): Directory of label data.
        save_dir (str): Directory to save results.
        if_show (bool): Whether to show the plot.

    Returns:
        None
    """
    # Find image and label files
    label_files = [file for file in os.listdir(label_dir) if file.endswith('.nii.gz')]
    cobb_angles_summary = []

    try:
        shutil.rmtree(save_dir)
        print(f"Folder '{save_dir}' has been deleted.")
    except FileNotFoundError:
        print(f"Folder '{save_dir}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    os.makedirs(save_dir, exist_ok=True)

    for label_file in label_files:
        # Construct corresponding image file path
        if not os.path.exists(os.path.join(img_dir, label_file)):
            img_file = label_file.replace('.nii.gz', '_0000.nii.gz')
        else:
            img_file = label_file        
        label_path = os.path.join(label_dir, label_file)
        img_path = os.path.join(img_dir, img_file)
        name = os.path.basename(label_file).replace('.nii.gz', "")
        
        # Read label data
        label = sitk.ReadImage(label_path)
        mask = sitk.GetArrayFromImage(label)
        img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img)
        
        # Get middle layer
        middle_index = mask.shape[0] // 2
        layer = mask[middle_index]
        cobb_angles = {}
        V = np.zeros(layer.shape, np.uint8)
        V[layer == 2] = 255
        num_V, V_labels, stats, _ = cv2.connectedComponentsWithStats(V, connectivity=8, ltype=None)

        # Filter out small regions with area < 50
        large_components = [i for i in range(1, num_V) if stats[i, cv2.CC_STAT_AREA] >= 50]
        
        # Only calculate if there are more than 6 large vertebrae
        if len(large_components) > 6:
            # Plot image and calculate Cobb angle
            fig = plt.figure(1)
            plt.imshow(img_array[middle_index], cmap='gray')  # Display middle layer image

            # Extract coordinates of vertebrae points for the first and sixth vertebrae
            V_label = np.zeros(layer.shape, np.uint8)
            V_label[V_labels == large_components[0]] = 255
            contours, _ = cv2.findContours(V_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cont = contours[0]
            epsilon = 0.04 * cv2.arcLength(cont, True)
            box = cv2.approxPolyDP(cont, epsilon, True)
            box = box.squeeze()
            sorted_box = box[np.lexsort(-box.T)]
            point2_1 = sorted_box[0]
            point2_2 = sorted_box[1]
            if point2_2[0] < point2_1[0]:
                point2_2 = sorted_box[2]
                
            V_label = np.zeros(layer.shape, np.uint8)
            V_label[V_labels == large_components[5]] = 255
            contours, _ = cv2.findContours(V_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cont = contours[0]
            epsilon = 0.04 * cv2.arcLength(cont, True)
            box = cv2.approxPolyDP(cont, epsilon, True)
            box = box.squeeze()
            sorted_box = box[np.lexsort(-box.T)]
            point7_1 = sorted_box[0]
            point7_2 = sorted_box[1]
            if point7_2[0] < point7_1[0]:
                point7_2 = sorted_box[2]

            # Plot and calculate Cobb angle
            plt.title(f'Cobb Angle for Middle Layer')
            fig = plot_cobb_angle(point2_1, point2_2, point7_1, point7_2, fig, True)
            cobb_angle = calculate_cobb_angle(point2_1, point2_2, point7_1, point7_2)
            cobb_angles[f'Cobb_Angle_C2_C7_Middle_Layer'] = round(cobb_angle, 2)
            fig.patch.set_alpha(0.0) 
            if len(save_dir) > 0:
                save_path = os.path.join(save_dir, name)
                fig.savefig(save_path + '_Cobb_Angle_Middle_Layer.png', dpi=300)  # Save image
            if if_show:
                plt.show()
            plt.close()

            # Save Cobb angle information to text file
            cobb_angles_summary.append((name, cobb_angle))

    # Save all Cobb angles to a summary text file
    summary_txt_path = os.path.join(save_dir, 'Cobb_Angles_Summary.txt')
    with open(summary_txt_path, 'w') as f:
        for name, angle in cobb_angles_summary:
            f.write(f'{name}: {angle:.2f} degrees\n')
    print(f'Saved all Cobb angles to {summary_txt_path}')
    
    # Save Cobb angles to an Excel file
    df = pd.DataFrame(cobb_angles_summary, columns=['Name', 'Cobb_Angle_C2_C7_Middle_Layer'])
    summary_excel_path = os.path.join(save_dir, 'Cobb_Angles_Summary.xlsx')
    df.to_excel(summary_excel_path, index=False)
    print(f'Saved all Cobb angles to {summary_excel_path}')
    
def main():
    parser = argparse.ArgumentParser(description='处理模型预测结果并生成诊断图')
    parser.add_argument('--main_folder', type=str, help='包含模型文件夹的根目录路径')
    parser.add_argument('--save_folder', type=str, help='保存结果的根目录路径')
    parser.add_argument('--if_show', type=bool,default=False, help='是否显示图像')
    args = parser.parse_args()
    
    img_path = os.path.join(args.main_folder, 'img')
    label_path = os.path.join(args.main_folder, 'pred_protrude')
    save_path = args.save_folder
    os.makedirs(save_path, exist_ok=True)
    
    process_patient_C27(img_path, label_path, save_path, args.if_show)
    
if __name__ == '__main__':
    main()
