# Import the libraries
import torch
from torch import Tensor
import torch.nn as nn



_all_blocks = ["_GeneratorResidualBlock", "_GeneratorUpsampleBlock", "_DiscriminatorBlock"]

class _GeneratorResidualBlock(nn.Module):
    def __init__(self, channels=64) -> None:
        super(_GeneratorResidualBlock, self).__init__()
        # Padding is made equal to 1 to ensure size of filters == size of input image
        self.conv1 = nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, stride =1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features = channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, stride =1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features = channels)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        y = torch.add(x, out)
        return y

class _GeneratorUpsampleBlock(nn.Module):
    def __init__(self, channels = 64, up_scale = 2) -> None:
        super(_GeneratorUpsampleBlock, self).__init__()
        # each Upsample block upsamples the image by up_scale.
        self.conv = nn.Conv2d(in_channels = channels, out_channels = channels * up_scale ** 2, kernel_size = 3, stride =1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)

        return out


class _DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels = 64 , output_channels = 128, stride = 1) -> None:
        super(_DiscriminatorBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels = input_channels, out_channels = output_channels, kernel_size = 3, stride = stride, padding=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.lrelu(out)

        return out



if __name__ == "__main__":
    # Cross-check if blocks are built correctly
    input1 = torch.rand((10, 64, 32, 32))
    input2 = torch.rand((10, 64, 32, 32))
    input3 = torch.rand((10, 64, 32, 32))

    model1 = _GeneratorResidualBlock()
    model2 = _GeneratorUpsampleBlock()
    model3 = _DiscriminatorBlock()

    out1 = model1(input1)
    out2 = model2(input2)
    out3 = model3(input3)

    print(out1.shape)
    print(out2.shape)
    print(out3.shape)