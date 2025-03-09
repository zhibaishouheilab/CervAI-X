"""
MRI Segmentation Analysis Tool

This tool processes MRI (Magnetic Resonance Imaging) data and corresponding segmentation masks 
to automatically detect and annotate protruded regions (e.g., intervertebral disc herniation). 
It generates visualized results with arrow markers and anatomical labels, saved as PNG files with transparent backgrounds.

Key Features:
1. Connected Component Analysis & Filtering:
   - Performs connected region detection on protruded areas
   - Filters components using anatomical labels (e.g., disc tags) and spatial positional criteria
   - Excludes non-adjacent regions through positional relationship analysis
   - Preserves only clinically significant protrusions

2. Automated Annotation & Visualization:
   - Adds red arrow annotations indicating protrusion direction
   - Labels anatomical structures (e.g., C2-C7 vertebrae) near target regions
   - Outputs PNG images with light gray background for optimal contrast
   - Maintains original image resolution (512×512) with transparent annotations

3. Advanced Features (2024/09/02 Update):
   - Implements spinal canal-side protrusion filtering
   - Records protrusion locations by intervertebral disc segment
   - Generates structured Excel reports with positional metadata
"""
import os
import SimpleITK as sitk
import numpy as np
import cv2
import pandas as pd
import argparse

def find_files(directory, extension):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))
    return file_list

def process_images(mr_dir, seg_dir, save_dir, arrow_label, text_label, ivd_label):
    mr_files = find_files(mr_dir, '.nii.gz')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results_list = []

    for mr_file in mr_files:
        name = mr_file.split('/')[-1]
        seg_file = os.path.join(seg_dir, name).replace("_0000", "")
        mr_img = sitk.ReadImage(mr_file)
        seg_img = sitk.ReadImage(seg_file)

        mr_array = sitk.GetArrayFromImage(mr_img)
        seg_array = sitk.GetArrayFromImage(seg_img)

        num_slices = seg_array.shape[0]
        for i in range(num_slices):
            mr_slice = mr_array[i]
            seg_slice = seg_array[i]

            mr_slice_normalized = ((mr_slice - np.min(mr_slice)) / (np.max(mr_slice) - np.min(mr_slice)) * 255).astype(np.uint8)

            seg_slice_arrow = (seg_slice == arrow_label).astype(np.uint8)
            seg_slice_ivd = (seg_slice == ivd_label).astype(np.uint8)
            seg_slice_text = (seg_slice == text_label).astype(np.uint8)

            combined_mask = seg_slice_arrow | seg_slice_ivd
            num_combined_features, labeled_combined_img = cv2.connectedComponents(combined_mask, connectivity=8)
            num_arrow_features, labeled_arrow_img = cv2.connectedComponents(seg_slice_arrow, connectivity=8)

            valid_arrow_mask = np.zeros_like(seg_slice_arrow)
            arrow_sizes = [0] * 5  # Initialize with 0 for arrow sizes in the 5 combined_mask regions

            arrow_status = [0] * 5  # Initialize with 0 for 5 combined_mask regions

            for arrow_component in range(1, num_arrow_features):
                arrow_component_mask = (labeled_arrow_img == arrow_component)
                arrow_coords = np.column_stack(np.where(arrow_component_mask))
                arrow_size = np.sum(arrow_component_mask)  # Calculate the size of the arrow component

                arrow_center_x, arrow_center_y = arrow_coords.mean(axis=0)
                combined_label_at_arrow = labeled_combined_img[int(arrow_center_x), int(arrow_center_y)]
                if combined_label_at_arrow == 0:
                    continue

                combined_component_mask = (labeled_combined_img == combined_label_at_arrow)
                combined_coords = np.column_stack(np.where(combined_component_mask))

                if arrow_center_y > combined_coords[:, 1].mean():
                    valid_arrow_mask[arrow_component_mask] = 1

                # Mark the corresponding combined_mask region and record arrow size
                    if combined_label_at_arrow <= 5:  # Only consider the first 5 regions
                        arrow_status[combined_label_at_arrow - 1] = 1
                        arrow_sizes[combined_label_at_arrow - 1] += arrow_size  # Accumulate sizes if multiple arrows are found in the same region

            # 提取文字标签的连通域
            num_features_text, labeled_img_text = cv2.connectedComponents(seg_slice_text, connectivity=8)
            
            output_img = cv2.cvtColor(mr_slice_normalized, cv2.COLOR_GRAY2RGB)

            # 标注有效的箭头连通域
            if np.any(valid_arrow_mask):
                num_features_valid_arrow, labeled_valid_arrow_img = cv2.connectedComponents(valid_arrow_mask, connectivity=8)
                for component in range(1, num_features_valid_arrow):
                    component_mask = labeled_valid_arrow_img == component
                    coords = np.column_stack(np.where(component_mask))
                    if coords.any():
                        y, x = coords.mean(axis=0).astype(int)
                        # 绘制红色箭头，从右上方向左下方45度指向label区域
                        arrow_tip = (x + 5, y)
                        arrow_tail = (x + 20, y - 15)
                        output_img = cv2.arrowedLine(output_img, arrow_tail, arrow_tip, (0, 0, 255), 2, tipLength=0.3)

            # 标注文字标签的连通域
            centroids = []
            for component in range(1, num_features_text):
                component_mask = labeled_img_text == component
                coords = np.column_stack(np.where(component_mask))
                
                # 过滤掉大小小于50的连通域
                if np.sum(component_mask) < 50:
                    continue

                if coords.any():
                    y, x = coords.mean(axis=0).astype(int)
                    centroids.append((y, x))

            # 按 y 坐标排序
            centroids.sort()
            labels = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7']
            
            for idx, (y, x) in enumerate(centroids):
                if idx < len(labels):
                    output_img = cv2.putText(output_img, labels[idx], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    
            # 保存图像
            slice_filename = os.path.join(save_dir, f'{os.path.basename(mr_file)}_slice_{i + 1}.png')
            cv2.imwrite(slice_filename, output_img)


            results_list.append({
                'File': name,
                'c2/3': arrow_status[0],
                'c2/3_size': arrow_sizes[0],
                'c3/4': arrow_status[1],
                'c3/4_size': arrow_sizes[1],
                'c4/5': arrow_status[2],
                'c4/5_size': arrow_sizes[2],
                'c5/6': arrow_status[3],
                'c5/6_size': arrow_sizes[3],
                'c6/7': arrow_status[4],
                'c6/7_size': arrow_sizes[4]
            })

    # Convert results to DataFrame and save to Excel
    df_results = pd.DataFrame(results_list)
    excel_save_path = os.path.join(save_dir, 'arrow_status.xlsx')
    df_results.to_excel(excel_save_path, index=False)
    print(f"Results saved to {excel_save_path}")

def main():
    parser = argparse.ArgumentParser(description='处理模型预测结果并生成诊断图')
    parser.add_argument('--main_folder', type=str, help='包含模型文件夹的根目录路径')
    parser.add_argument('--save_folder', type=str, help='保存结果的根目录路径')
    parser.add_argument('--arrow_label', type=int, default=5, help='突出部分的标签值')
    parser.add_argument('--text_label', type=int, default=2, help='解剖结构的标签值')
    parser.add_argument('--ivd_label', type=int, default=1, help='椎间盘的标签值')
    args = parser.parse_args()
    
    mr_dir = os.path.join(args.main_folder, 'img')
    seg_dir = os.path.join(args.main_folder, 'pred_protrude')
    save_dir = args.save_folder
    os.makedirs(save_dir, exist_ok=True)
    
    process_images(mr_dir, seg_dir, save_dir, args.arrow_label, args.text_label, args.ivd_label)
    
if __name__ == '__main__':
    main()