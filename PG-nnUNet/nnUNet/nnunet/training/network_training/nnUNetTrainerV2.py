#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# PG-nnUNet

from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
import torch.nn.functional as F
from scipy.ndimage import label, center_of_mass
import time


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2, reduction="mean"):
    """
    Focal Loss for keypoint detection tasks
    :param inputs: predicted heatmaps (B, 1, H, W)
    :param targets: ground truth heatmaps (B, 1, H, W)
    :param alpha: balancing parameter for class imbalance
    :param gamma: focusing parameter to down-weight easy examples
    :param reduction: reduction method ('mean' or 'sum')
    :return: focal loss value
    """
    # Apply sigmoid activation to predictions
    p = torch.sigmoid(inputs)
    
    # Calculate binary cross entropy
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    
    # Calculate p_t
    p_t = p * targets + (1 - p) * (1 - targets)
    
    # Calculate focal loss
    loss = ce_loss * ((1 - p_t) ** gamma)
    
    # Apply alpha balancing
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    # Apply reduction
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

def compute_focal_loss(heatmaps, target, loss_weights, alpha=0.25, gamma=2):
    """
    计算多个层次的预测热力图与GT热力图的Focal Loss
    :param heatmaps: 多个层次的预测热力图列表，每个元素形状为 (B, 1, H, W)
    :param target: 输入的目标热力图，形状为 (B, 1, H, W)
    :param loss_weights: 各层损失的权重
    :param alpha: Focal Loss的alpha参数
    :param gamma: Focal Loss的gamma参数
    :return: 返回加权的Focal Loss
    """
    total_focal_loss = 0
    
    for i, heatmap in enumerate(heatmaps):
        # 对GT热力图进行降采样，以匹配当前预测热力图的大小
        gt_heatmap_resized = F.interpolate(target, size=heatmap.shape[2:], mode='bilinear', align_corners=False)
        gt_heatmap_resized = to_cuda(gt_heatmap_resized)
        
        # 打印调试信息
        print(f"Layer {i} Debug:")
        print(f"  Output heatmap shape: {heatmap.shape}")
        print(f"  Output min: {heatmap.min().item():.4f}, max: {heatmap.max().item():.4f}, mean: {heatmap.mean().item():.4f}")
        print(f"  Target min: {gt_heatmap_resized.min().item():.4f}, max: {gt_heatmap_resized.max().item():.4f}, mean: {gt_heatmap_resized.mean().item():.4f}")
        
        # 计算最大值位置（正确的argmax使用方式）
        with torch.no_grad():
            # 输出热力图的最大值位置
            flat_idx_out = torch.argmax(heatmap.view(heatmap.size(0), -1), dim=1)
            coords_out = torch.stack([flat_idx_out // heatmap.size(3), flat_idx_out % heatmap.size(3)], dim=1)
            
            # 目标热力图的最大值位置
            flat_idx_gt = torch.argmax(gt_heatmap_resized.view(gt_heatmap_resized.size(0), -1), dim=1)
            coords_gt = torch.stack([flat_idx_gt // gt_heatmap_resized.size(3), flat_idx_gt % gt_heatmap_resized.size(3)], dim=1)
            
            # 计算最大值的距离（以像素为单位）
            distances = torch.norm(coords_out.float() - coords_gt.float(), dim=1)
            print(f"  Avg distance: {distances.mean().item():.2f} pixels")
        
        # 计算Focal Loss
        focal_loss = sigmoid_focal_loss(heatmap, gt_heatmap_resized, alpha, gamma, reduction="mean")
        print(f"  Focal loss: {focal_loss.item():.4f}")
        total_focal_loss += loss_weights[i] * focal_loss
    
    return total_focal_loss


def compute_mse_loss(heatmaps, target, loss_weights):
    """
    计算多个层次的预测热力图与GT热力图的MSE损失
    :param heatmaps: 多个层次的预测热力图列表，每个元素形状为 (B, C, H, W)
    :param target: 输入的目标图像，形状为 (B, 1, H, W)
    :param target_label: 用于生成热力图的标签值
    :param sigma_scale: 用于计算sigma的比例系数
    :return: 返回平均的MSE损失
    """
    # 生成原始分辨率的GT_heatmap
    #target = target.squeeze(1).long()  # 转换为整数类型
    #gt_heatmap = generate_gt_heatmap(target, sigma_scale, min_region_size=10)
    gt_heatmap = target
    #print(target.max())
    
    total_mse_loss = 0
    
    for i, heatmap in enumerate(heatmaps):
        #print(heatmap.max(), heatmap.min())
        # 对GT_heatmap进行降采样，以匹配当前预测热力图的大小
        #print(heatmap.shape,gt_heatmap.shape) torch.Size([14, 1, 512, 448]) torch.Size([14, 1, 512, 448])
        gt_heatmap_resized = F.interpolate(gt_heatmap, size=heatmap.shape[2:], mode='bilinear', align_corners=False)
        gt_heatmap_resized = to_cuda(gt_heatmap_resized)
        #print(gt_heatmap_resized.shape,heatmap.shape)
        
        # 计算MSE损失
        mse_loss = F.mse_loss(heatmap, gt_heatmap_resized)
        total_mse_loss += loss_weights[i]*mse_loss
    
    # 计算所有层次的MSE损失的平均值
    #average_mse_loss = total_mse_loss / len(heatmaps)
    return total_mse_loss

class nnUNetTrainerV2(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 300
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.heatmap_loss_weight = 1.0  # 热图损失的权重
        self.edge_loss_weight = 0.2    # 边缘损失的权重
        self.pin_memory = True
        self.focal_alpha = 0.25        # Focal Loss的alpha参数
        self.focal_gamma = 2           # Focal Loss的gamma参数

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            
            # now wrap the loss
            # 使用自定义损失函数
            self.loss = self.custom_loss
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                #print(self.dl_tr.keys())
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                    #heatmap_channels=56  # 添加热图通道参数
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                                    #det_output_channels=56
                                    )
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        
    def custom_loss(self, output_seg, output_heatmap, target_seg, target_heatmap):
        """
        自定义损失函数，结合分割损失、热图损失和边缘损失
        """
        # 分割损失 (深度监督)
        seg_loss = 0
        for i in range(len(output_seg)):
            seg_loss += self.ds_loss_weights[i] * F.cross_entropy(output_seg[i], target_seg[i].squeeze(1).long())
        
        # 热图损失 (MSE)
        #heatmap_loss = compute_mse_loss(output_heatmap, target_heatmap, self.ds_loss_weights)
        # 热图损失 (Focal Loss)
        heatmap_loss = compute_focal_loss(output_heatmap, target_heatmap, self.ds_loss_weights, 
                                         alpha=self.focal_alpha, gamma=self.focal_gamma)
        
        # 组合损失
        total_loss = seg_loss + self.heatmap_loss_weight * heatmap_loss 
        print("seg_loss:", seg_loss.item(), "heatmap_loss:", heatmap_loss.item(), "total_loss:", total_loss.item())
        
        return total_loss

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0]
        output = output[0]
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        #print("data_dict keys:", data_dict.keys())
        data = data_dict['data']
        target_seg = data_dict['target']  # 分割目标
        target_heatmap = data_dict['heatmap']  # 热图目标
        
        
        #print("data shape:", data.shape)
        #print(target_seg)
        #print("target_seg shape:", len(target_seg), target_seg[0].shape)
        #print("target_heatmap shape:", target_heatmap.shape)

        data = maybe_to_torch(data)
        target_seg = maybe_to_torch(target_seg)
        target_heatmap = maybe_to_torch(target_heatmap)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target_seg = to_cuda(target_seg)
            target_heatmap = to_cuda(target_heatmap)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                # 网络输出两个结果：分割和热图
                output_seg, output_heatmap = self.network(data)
                #print("output_seg shape:", len(output_seg), output_seg[0].shape)
                #print("output_heatmap shape:", len(output_heatmap), output_heatmap[0].shape)

                # 计算自定义损失
                l = self.custom_loss(output_seg, output_heatmap, target_seg, target_heatmap)
                
                
                #self.run_online_evaluation(output_seg, target_seg)
                # 在线评估（仅使用分割输出）
                if run_online_evaluation:
                    self.run_online_evaluation(output_seg, target_seg)
                

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output_seg, output_heatmap = self.network(data)
            l = self.custom_loss(output_seg, output_heatmap, target_seg, target_heatmap)
            
            if run_online_evaluation:
                self.run_online_evaluation(output_seg[-1], target_seg[-1])
                
            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        del data, target_seg, target_heatmap
        return l.detach().cpu().numpy()

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
