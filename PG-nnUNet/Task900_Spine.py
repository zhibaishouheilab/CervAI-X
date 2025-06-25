import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
#import sys # 用于接受输入参数，表示任务号，如“Task900_Spine”
#确定一下有几个label，是否需要修改

if __name__ == '__main__':
    base = "/home/ubuntu/Project/nnUNet/dataset/nnUNet_raw_data_base/nnUNet_raw_data/"#数据保存的上级文件夹
    # this folder should have the training and testingr subfolders

    # now start the conversion to nnU-Net:
    task_name = 'Task902_spine'
    target_base = join(base, task_name)#nnUNet_raw_data会由paths和给出的base确定
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")#实际上没有test
    target_labelsTs = join(target_base, "labelsTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    # finally we can call the utility for generating a dataset.json
    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('1'),#modality模态，这里只有一个
                          labels={0: 'background', 1: 'IVD',2:'V',3:'CSV',4:'SC'}, dataset_name=task_name, license='hands off!')
