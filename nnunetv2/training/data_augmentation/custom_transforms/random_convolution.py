import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class RandomConvolutionTransform(AbstractTransform):
    def __init__(
        self,
        n_hidden_layers: int,
        k: int,
        h_hidden_channels: int = 32,
        n_modalities: int = 1,
        non_linearity: bool = False,
    ):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv3d(
                in_channels=n_modalities,
                out_channels=h_hidden_channels,
                kernel_size=k,
                stride=1,
                padding="same",
                groups=n_modalities,
            )
        )

        for _ in range(1, n_hidden_layers - 1):
            self.conv_layers.append(
                nn.Conv3d(
                    in_channels=h_hidden_channels,
                    out_channels=h_hidden_channels,
                    kernel_size=k,
                    stride=1,
                    padding="same",
                    groups=n_modalities,
                )
            )
            if non_linearity:
                self.conv_layers.append(nn.LeakyReLU())

        self.img_out = nn.Conv3d(
            in_channels=h_hidden_channels,
            out_channels=n_modalities,
            groups=n_modalities,
            kernel_size=1,
            stride=1,
            padding="same",
        )

        self.conv_layers.requires_grad_(False)
        self.p = 0.5

    def __call__(self, **data_dict: torch.Tensor) -> dict[str, torch.Tensor]:

        if np.random.rand() < self.p:
            return data_dict
        # randomize the weights of the conv layers
        self.init_weights()

        x: torch.Tensor = data_dict["image"][None]

        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.img_out(x)
        x = x.squeeze(0)

        # renormalize with z_norm per channel (0th dim)
        x = (x - x.mean(dim=(tuple(range(1, x.ndim))), keepdim=True)) / (
            x.std(dim=(tuple(range(1, x.ndim))), keepdim=True) + 1e-5
        )

        # # for vis purposes, noramlize each volume to [0, 1]
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
