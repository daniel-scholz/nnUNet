from typing import List, Tuple, Union

import numpy as np
from batchgenerators.transforms.abstract_transforms import Compose
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from nnunetv2.training.data_augmentation.custom_transforms.random_convolution import (
    RandomConvolutionTransform,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerRC(nnUNetTrainer):
    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        transform = nnUNetTrainer.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm,
            is_cascaded,
            foreground_labels,
            regions,
            ignore_label,
        )
        rc = RandomConvolutionTransform(
            n_hidden_layers=4, k=3, n_modalities=2, non_linearity=False
        )
        transform = Compose([transform, rc])
        return transform  # type: ignore
