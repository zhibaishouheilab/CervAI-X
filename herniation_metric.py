"""
20240912
对多个患者椎间盘突出大小的 Excel 文件（disc_files）进行分析，并与唯一的人工标注真值文件（true_file）进行对比。代码通过设定固定的阈值（默认值为0）来判断椎间盘是否突出（标记为1或0），并计算以下指标：

准确率（Accuracy）
精确率（Precision）
召回率（Recall）
F1指数（F1 Score）
混淆矩阵的四个值（真阳性、假阳性、真阴性、假阴性）
所有分析结果会记录到一个文本文件中，便于后续查看和评估
"""
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 根据阈值标记突出（大于阈值为1，其他为0）
def apply_threshold(disc_df, threshold=0):
    disc_pred = disc_df.iloc[:, 1:].applymap(lambda x: 1 if x > threshold else 0)  # 跳过患者名称列
    return disc_pred

# 计算各项指标以及混淆矩阵的值
def calculate_metrics(disc_pred, true_df):
    y_true = true_df.iloc[:, 1:].values.flatten()  # 将真值展开成一维数组
    y_pred = disc_pred.values.flatten()  # 将预测结果展开成一维数组

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  # 混淆矩阵的四个值
    
    return accuracy, precision, recall, f1, tn, fp, fn, tp

# 保存结果到文本文件
def save_results_to_file(file_name, results):
    with open(file_name, 'a') as f:
        f.write("File: " + results['File'] + "\n")
        f.write(f"Accuracy: {results['Accuracy']:.4f}, Precision: {results['Precision']:.4f}, "
                f"Recall: {results['Recall']:.4f}, F1 Score: {results['F1 Score']:.4f}\n")
        f.write(f"Confusion Matrix: TP={results['TP']}, FP={results['FP']}, TN={results['TN']}, FN={results['FN']}\n\n")

# 处理单个文件，计算指标
def process_file(disc_file, true_file, threshold=0):
    disc_df = pd.read_excel(disc_file)
    true_df = pd.read_excel(true_file, sheet_name='Sheet2')
    
    disc_pred = apply_threshold(disc_df, threshold)
    accuracy, precision, recall, f1, tn, fp, fn, tp = calculate_metrics(disc_pred, true_df)
    
    results = {
        'File': disc_file,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp
    }
    
    return results

# 主函数：处理多个disc文件和一个true文件
def main(disc_files, true_file, output_file):
    for disc_file in disc_files:
        results = process_file(disc_file, true_file)
        save_results_to_file(output_file, results)


# 调用函数
disc_files = ['900_size.xlsx','901_size.xlsx','902_size.xlsx','nested_size.xlsx', 'trans_size.xlsx']  # 示例，您可以添加多个文件
true_files = '椎间盘突出人工判断.xlsx'
output_file = '阈值结果.txt'

main(disc_files, true_files, output_file)

print(f"所有文件处理完毕，结果已保存到 {output_file}")
