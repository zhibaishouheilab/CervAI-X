"""
revised 2024/07/06
This script performs protrusion segmentation on MRI images of the cervical spine.
It extracts the segmentation of IVD protrusions and saves the results in the specified output path.

Functionality:
- Segment the IVD protrusions (protrude_seg)
- Save the segmented mask as a new image

Usage:
- Define the input and output paths.
- Run the script to process all the MRI images in the input path.
"""

import cv2
import numpy as np
import SimpleITK as sitk
import skimage
import math
import os
from pathlib import Path
import argparse

def connectedComponents_locate(img, size=50):
    """
    Locate connected components in the image and filter out small components.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    sorted_indices = np.argsort(centroids[1:, 1])
    sorted_indices = sorted_indices[stats[1:][sorted_indices][:, cv2.CC_STAT_AREA] >= size]
    num_labels = len(sorted_indices)
    new_labels = np.zeros_like(labels)
    new_stats = np.zeros([num_labels+1, 5])
    new_centroids = np.zeros([num_labels+1, 2])
    new_stats[0] = stats[0]
    new_centroids[0] = centroids[0]

    for new_label, old_label in enumerate(sorted_indices):
        new_labels[labels == old_label+1] = new_label+1
    new_stats[1:] = stats[1:][sorted_indices]
    new_centroids[1:] = centroids[1:][sorted_indices]
    return num_labels, new_labels, new_stats, new_centroids

def get_labels(mask):
    """
    Extract individual labels from the mask.
    """
    V = np.zeros(mask.shape, np.uint8)
    V[mask == 2] = 255
    IVD = np.zeros(mask.shape, np.uint8)
    IVD[mask == 1] = 255
    CSF = np.zeros(mask.shape, np.uint8)
    CSF[mask == 4] = 255
    SC = np.zeros(mask.shape, np.uint8)
    SC[mask == 3] = 255
    return IVD, V, SC, CSF

def protrude_seg(ori_mask, mask_name, slice):
    """
    Segment the IVD protrusions in the MRI image.
    """
    mask = ori_mask.copy()
    onlyseg_mask = ori_mask.copy()
    mask_slice = slice
    indice_max = ori_mask.max()
    seg_indice = indice_max + 1
    seg_line = indice_max + 2
    seg_dot = indice_max + 3

    IVD, V, SC, CSF = get_labels(mask)
    V_num_labels, V_labels, V_stats, V_centroids = connectedComponents_locate(V, size=50)
    IVD_num_labels, IVD_labels, IVD_states, IVD_centroids = connectedComponents_locate(IVD, size=50)

    dot_rightup = []
    dot_rightdown = []
    dot_leftup = []
    dot_leftdown = []

    if V_num_labels <= 6:
        return ori_mask

    for i in range(6):
        label = np.zeros(V.shape, np.uint8)
        label[V_labels == i+1] = 255
        contours, _ = cv2.findContours(label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cont = contours[0]
        epsilon = 0.02*cv2.arcLength(cont, True)
        box = cv2.approxPolyDP(cont, epsilon, True)
        dot_up = []
        dot_down = []

        for j in range(len(box)):
            if box[j][0][1] < V_centroids[i+1][1]:
                dot_down.append(box[j][0])
            else:
                dot_up.append(box[j][0])

        dot_down = [tuple(x) for x in dot_down]
        dot_up = [tuple(x) for x in dot_up]
        dot_down = sorted(dot_down)
        dot_up = sorted(dot_up)

        if not dot_down or not dot_up:
            print(f"error happened in {mask_slice+1}th slice, {i+1}th V")
            print('dot_down', dot_down)
            print('dot_up', dot_up)
            print(V_stats)

        dot_rightup.append(dot_up[-1])
        dot_leftup.append(dot_up[0])
        dot_rightdown.append(dot_down[-1])
        dot_leftdown.append(dot_down[0])

    for i in range(5):
        cv2.line(IVD, dot_rightup[i], dot_rightdown[i+1], (0, 0, 0), 4)
        cv2.line(IVD, dot_leftup[i], dot_leftdown[i+1], (0, 0, 0), 4)
        rr, cc = skimage.draw.ellipse(dot_rightup[i][0], dot_rightup[i][1], 2, 2)
        mask[cc, rr] = seg_dot
        rr, cc = skimage.draw.ellipse(dot_rightdown[i][0], dot_rightdown[i][1], 2, 2)
        mask[cc, rr] = seg_dot
        rr, cc = skimage.draw.ellipse(dot_leftup[i][0], dot_leftup[i][1], 2, 2)
        mask[cc, rr] = seg_dot
        rr, cc = skimage.draw.ellipse(dot_leftdown[i][0], dot_leftdown[i][1], 2, 2)
        mask[cc, rr] = seg_dot
        rr, cc = skimage.draw.line(dot_rightup[i][0], dot_rightup[i][1], dot_rightdown[i+1][0], dot_rightdown[i+1][1])
        mask[cc, rr] = seg_line
        rr, cc = skimage.draw.line(dot_leftup[i][0], dot_leftup[i][1], dot_leftdown[i+1][0], dot_leftdown[i+1][1])
        mask[cc, rr] = seg_line
    
    for i in range(IVD_num_labels-1):
        IVD_label = IVD.copy()
        IVD_label[IVD_labels != i+1] = 0
        Number, Labels, Stats, Centroids = cv2.connectedComponentsWithStats(IVD_label)
        label_size = Stats[:, 4]
        sorted_label_size = sorted(enumerate(label_size), key=lambda label_size: label_size[1])
        sorted_index = [x[0] for x in sorted_label_size]
        if Number >= 3:
            for j in range(Number-2):
                mask[Labels == sorted_index[j]] = seg_indice
    onlyseg_mask[mask == seg_indice] = seg_indice
    
    #return mask, onlyseg_mask
    return onlyseg_mask

def diag_allslice(mask_path, type, output_path):
    """
    Perform diagnostic operations on all slices.
    """
    mask_name = mask_path.name
    mask_path = str(mask_path)
    #print(isinstance(mask_path, str))
    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)
    temp_mask = mask_array.copy()
    prod_mask = mask_array.copy()
    h, x, y = mask_array.shape

    segPath = output_path['prod']

    for i in range(h):
        print(f"Now diagnose slice {i+1}")
        if 'prod' in type:
            prod_seg = protrude_seg(temp_mask[i], mask_name, i)
            prod_mask[i] = prod_seg

    prod_mask = sitk.GetImageFromArray(prod_mask)
    prod_mask.SetDirection(mask.GetDirection())
    prod_mask.SetSpacing(mask.GetSpacing())
    prod_mask.SetOrigin(mask.GetOrigin())
    sitk.WriteImage(prod_mask, os.path.join(segPath, mask_name))
    
    
def main():
    parser = argparse.ArgumentParser(description='Segment protrusion')
    parser.add_argument('--main_folder', type=str, help='包含模型文件夹的根目录路径')
    parser.add_argument('--save_folder', type=str, help='保存结果的根目录路径')
    args = parser.parse_args()
    
    main_folder = args.main_folder
    save_folder = args.save_folder
    
    gt_folder_path = os.path.join(main_folder,'gt')
    pred_folder_path = os.path.join(main_folder,'pred')
    
    gt_file = Path(gt_folder_path)
    gt_file_path = sorted(gt_file.glob('*.nii.gz'))
    gt_save_path = os.path.join(save_folder,'gt_protrude')
    Path(gt_save_path).mkdir(parents=True, exist_ok=True)
    
    for idx in range(len(gt_file_path)):
        gt = gt_file_path[idx]
        gt_name = gt.name
        
        print(f"now is gt:{gt_name}")
        gt_output_path = {'prod': gt_save_path}
        diag_allslice(gt_file_path[idx], ['prod'], gt_output_path)
        
    pred_file = Path(pred_folder_path)
    pred_file_path = sorted(pred_file.glob('*.nii.gz'))
    pred_save_path = os.path.join(save_folder,'pred_protrude')
    Path(pred_save_path).mkdir(parents=True, exist_ok=True)
    
    for idx in range(len(pred_file_path)):
        mask = pred_file_path[idx]
        mask_name = mask.name
        
        print(f"now is pred:{mask_name}")
        pred_output_path = {'prod': pred_save_path}
        diag_allslice(pred_file_path[idx], ['prod'], pred_output_path)

if __name__ == '__main__':
    main()
