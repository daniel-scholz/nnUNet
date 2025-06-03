import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class RandomConvolutionTransform(BasicTransform):
    def __init__(
        self,
        n_hidden_layers: int,
        k: int,
        h_hidden_channels: int = 4,
        updown_sampling: bool = False,
        non_linearity: bool = False,
    ):
        super().__init__()
        self.conv_layers = None
        self.n_hidden_layers = n_hidden_layers
        self.k = k
        self.n_hidden_channels = h_hidden_channels
        self.non_linearity = non_linearity
        self.updown_sampling = updown_sampling
        self.target_size = 512

    def __call__(self, **data_dict: torch.Tensor) -> dict[str, torch.Tensor]:

        # randomize the weights of the conv layers

        x: torch.Tensor = data_dict["image"]
        if self.conv_layers is None:
            self.init_conv_layers(n_modalities=x.shape[0])

        self.init_weights()

        x = x.unsqueeze(0)  # add batch dimension

        # save original mean and std for each channel
        mean = x.mean(dim=(tuple(range(1, x.ndim))), keepdim=True)
        std = x.std(dim=(tuple(range(1, x.ndim))), keepdim=True) + 1e-5

        if self.updown_sampling:
            # upsample to longest size == self.target_size
            target_size = self.target_size
            current_max_size = max(
                x.shape[2:]
            )  # assuming x is of shape (B, C, D, H, W)
            scale = target_size / current_max_size
            new_size = tuple(
                int(dim * scale) for dim in x.shape[2:]
            )  # calculate new size for D, H, W
            x = nn.functional.interpolate(
                x,
                size=new_size,
                mode="trilinear",
                align_corners=True,
            )

        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.img_out(x)

        # resize to original size if upsampling was done
        if self.updown_sampling:
            original_size = data_dict["image"].shape[1:]
            x = nn.functional.interpolate(
                x,
                size=original_size,
                mode="trilinear",
                align_corners=True,
            )

        # renormalize to previous mean and std
        x = (x - x.mean(dim=(tuple(range(1, x.ndim))), keepdim=True)) / (
            x.std(dim=(tuple(range(1, x.ndim))), keepdim=True) + 1e-5
        )
        x = x * std + mean

        x = x.squeeze(0)

        # for vis purposes, noramlize each volume to [0, 1]
        # x_vis = torch.stack(
        #     [(x_ - x_.min()) / (x_.max() - x_.min()) for x_ in x], dim=0
        # )

        # # Visualize the middle slice of the last dimension for each channel in the batch
        # mid_slice = x_vis.shape[-1] // 2
        # # x_vis: ( C, D, H, W)
        # # We'll take the middle slice along D (last dimension)
        # # Resulting shape: (B, C, H, W)
        # mid_slices = x_vis[..., mid_slice]
        # # save all images to disk
        # for i in range(mid_slices.shape[0]):

        #     plt.imsave(
        #         f"slice_{i}.png",
        #         mid_slices[i].cpu().numpy(),
        #         cmap="gray",
        #     )

        data_dict["image"] = x
        return data_dict

    def init_conv_layers(self, n_modalities: int):

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv3d(
                in_channels=n_modalities,
                out_channels=self.n_hidden_channels,
                kernel_size=self.k,
                stride=1,
                padding="same",
                groups=n_modalities,
            )
        )

        for _ in range(1, self.n_hidden_layers - 1):
            self.conv_layers.append(
                nn.Conv3d(
                    in_channels=self.n_hidden_channels,
                    out_channels=self.n_hidden_channels,
                    kernel_size=self.k,
                    stride=1,
                    padding="same",
                    groups=n_modalities,
                )
            )
            if self.non_linearity:
                self.conv_layers.append(nn.LeakyReLU())

        self.img_out = nn.Conv3d(
            in_channels=self.n_hidden_channels,
            out_channels=n_modalities,
            groups=n_modalities,
            kernel_size=1,
            stride=1,
            padding="same",
        )

        self.conv_layers.requires_grad_(False)

    def init_weights(self):
        for conv_layer in self.conv_layers:
            if isinstance(conv_layer, nn.Conv3d):
                torch.nn.init.kaiming_normal_(conv_layer.weight)

        torch.nn.init.kaiming_normal_(self.img_out.weight)


if __name__ == "__main__":
    # Test the transform
    def main():
        n_modalities = 1
        transform = RandomConvolutionTransform(
            n_hidden_layers=3, k=3, h_hidden_channels=4, n_modalities=n_modalities
        )
        x = torch.randn(
            n_modalities, 64, 64, 64
        )  # Example input tensor (single sample, not batch)
        data_dict = {"image": x}
        transformed_data = transform(**data_dict)
        print(transformed_data["image"].shape)  # Should be the same shape as input

    main()
