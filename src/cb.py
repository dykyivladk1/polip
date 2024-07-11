import torch
import torch.nn as nn


class PixelNormLayer(nn.Module):
    """
    A custom layer for pixel normalization.

    This layer normalizes each pixel in the input tensor to have a unit variance.

    Parameters:
    epsilon (float): A small value to avoid division by zero. Default is 1e-8.
    """
    def __init__(self, epsilon=1e-8):
        """
        Initialize the PixelNormLayer.

        Parameters:
        epsilon (float): A small value to avoid division by zero. Default is 1e-8.
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
    


class UpSample(nn.Module):
    """
    An upsampling layer that increases the spatial dimensions of the input using nearest neighbor upsampling
    followed by a convolutional layer.

    Parameters:
    in_channels (int): Number of input and output channels.

    Methods:
    forward(x): Applies the upsampling operations to the input tensor x.
    """
    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, *args):
        return self.upsample(x)




class DownSample(nn.Module):
    """
    A downsampling layer that reduces the spatial dimensions of the input using a convolutional layer.

    Parameters:
    channels (int): Number of input and output channels.

    Methods:
    forward(x): Applies the downsampling convolution to the input tensor x.
    """
    def __init__(self, channels):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, *args):
        return self.downsample(x)



