import torch
from torch import nn

# The first layer is the PixelCNN layer. This layer simply
# builds on the 2D convolutional layer, but includes masking.
class PixelConvLayer(nn.Conv2d):
    """
    Implementation of Masked CNN Class as explained in A Oord et. al. 
    Taken from https://github.com/jzbontar/pixelcnn-pytorch
	  """
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, padding=0, stride=1, padding_mode='zeros', dilation=1, groups=1, bias=True, device=None, dtype=None):
        super(PixelConvLayer, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        assert mask_type in {'A', 'B'}

        self.register_buffer('mask', self.weight.data.clone())
        _, _, height, width = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, height // 2, width // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, height // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(PixelConvLayer, self).forward(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=1
        )

        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.relu = nn.ReLU()

        self.pixelconv = PixelConvLayer(
            in_channels=in_channels // 2,
            out_channels=in_channels // 2,
            kernel_size=3,
            mask_type="B",
            padding="same")
        self.bn2 = nn.BatchNorm2d(in_channels // 2)
        self.conv2 = nn.Conv2d(
            in_channels=in_channels // 2,
            out_channels=in_channels,
            kernel_size=1
        )
        self.bn3 = nn.BatchNorm2d(in_channels)

    def forward(self, inputs):
        x = self.relu(self.bn1(self.conv1(inputs)))
        x = self.relu(self.bn2(self.pixelconv(x)))
        x = self.relu(self.bn3(self.conv2(x)))
        return inputs + x


class PixelCNN(nn.Module):
    def __init__(self, channels, num_residual_blocks, num_pixelcnn_layers, K, **kwargs):
        super(PixelCNN, self).__init__(**kwargs)

        modules = []
        modules.append(
            nn.Sequential(
                PixelConvLayer(
                    mask_type="A",
                    in_channels=K,
                    out_channels=channels,
                    kernel_size=7, padding="same"
                ),
                nn.BatchNorm2d(channels), nn.ReLU()
            ))

        for _ in range(num_pixelcnn_layers):
            modules.append(nn.Sequential(PixelConvLayer(mask_type='B',
                                                        in_channels=channels,
                                                        out_channels=channels,
                                                        kernel_size=3,
                                                        padding="same"),
                                         nn.BatchNorm2d(channels),
                                         nn.ReLU()))

        for _ in range(num_residual_blocks):
            modules.append(ResidualBlock(in_channels=channels))

        modules.append(
            nn.Conv2d(in_channels=channels, out_channels=K, kernel_size=1, padding="valid")
        )
        self.model = nn.Sequential(*modules)

    def forward(self, input):
        return self.model(input)

    def save(self, path="./pixelcnn_model.ckpt"):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))