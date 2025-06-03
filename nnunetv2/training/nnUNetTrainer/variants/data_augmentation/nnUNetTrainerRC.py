from typing import List, Tuple, Union

import numpy as np
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from nnunetv2.training.data_augmentation.custom_transforms.random_convolution import (
    RandomConvolutionTransform,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerRC(nnUNetTrainer):

    n_modalities: int = 2
    updown_sampling: bool = False

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

        transform: ComposeTransforms = nnUNetTrainer.get_training_transforms(  # type: ignore
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
            n_hidden_layers=2,
            k=3,
            non_linearity=False,
            updown_sampling=__class__.updown_sampling,
        )
        transform = ComposeTransforms(
            [*transform.transforms, RandomTransform(rc, apply_probability=0.5)]
        )
        return transform  # type: ignore
