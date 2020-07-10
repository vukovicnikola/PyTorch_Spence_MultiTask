import torch
import torch.nn as nn


# CoordConv Trick

class AddCoordinates(object):

    """Coordinate Adder Module as defined in 'An Intriguing Failing of
    Convolutional Neural Networks and the CoordConv Solution'
    (https://arxiv.org/pdf/1807.03247.pdf).
    This module concatenates coordinate information (`x`, `y`, and `r`) with
    given input tensor.
    `x` and `y` coordinates are scaled to `[-1, 1]` range where origin is the
    center. `r` is the Euclidean distance from the center and is scaled to
    `[0, 1]`.
    Args:
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`
    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, (C_{in} + 2) or (C_{in} + 3), H_{in}, W_{in})`
    Examples:
        >>> coord_adder = AddCoordinates(True)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_adder(input)
    """

    def __call__(self, image):
        batch_size, _, image_height, image_width = image.size()

        y_coords = 2.0 * torch.arange(image_height).unsqueeze(
            1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
        x_coords = 2.0 * torch.arange(image_width).unsqueeze(
            0).expand(image_height, image_width) / (image_width - 1.0) - 1.0

        coords = torch.stack((y_coords, x_coords), dim=0)

        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)

        image = torch.cat((coords.to(image.device), image), dim=1)

        return image


class CoordConv(nn.Module):

    """2D Convolution Module Using Extra Coordinate Information as defined
    in 'An Intriguing Failing of Convolutional Neural Networks and the
    CoordConv Solution' (https://arxiv.org/pdf/1807.03247.pdf).
    Args:
        Same as `torch.nn.Conv2d` with two additional arguments
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`
    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, C_{out}, H_{out}, W_{out})`
    Examples:
        >>> coord_conv = CoordConv(3, 16, 3, with_r=True)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_conv(input)
        >>> coord_conv = CoordConv(3, 16, 3, with_r=True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_conv(input)
        >>> device = torch.device("cuda:0")
        >>> coord_conv = CoordConv(3, 16, 3, with_r=True).to(device)
        >>> input = torch.randn(8, 3, 64, 64).to(device)
        >>> output = coord_conv(input)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 with_r=False):
        super(CoordConv, self).__init__()

        self.coord_adder = AddCoordinates()

        self.conv_layer = nn.Conv2d(in_channels, out_channels,
                                    kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    groups=groups, bias=bias)

    def forward(self, x):
        x = self.coord_adder(x)
        x = self.conv_layer(x)

        return x

# End CoordConv


def conv(n_inputs, n_filters, kernel_size=3, stride=1, bias=False) -> torch.nn.Conv2d:
    """Creates a convolution layer for `XResNet`."""
    return nn.Conv2d(n_inputs, n_filters,
                     kernel_size=kernel_size, stride=stride,
                     padding=kernel_size//2, bias=bias)


def conv_layer(n_inputs: int, n_filters: int,
               kernel_size: int = 3, stride=1,
               zero_batch_norm: bool = False, use_activation: bool = True,
               activation: torch.nn.Module = nn.ReLU(inplace=True),
               coordconv: bool = False) -> torch.nn.Sequential:
    """Creates a convolution block for `XResNet`."""

    if coordconv:  # use CoordConv
        # two extra channels are added with XY locations
        conv_2d = CoordConv(n_inputs+2, n_filters+2, kernel_size, stride=stride)
        batch_norm = nn.BatchNorm2d(n_filters+2)
    else:  # use regular conv
        conv_2d = conv(n_inputs, n_filters, kernel_size, stride=stride)
        batch_norm = nn.BatchNorm2d(n_filters)
    # initialize batch normalization to 0 if its the final conv layer
    nn.init.constant_(batch_norm.weight, 0. if zero_batch_norm else 1.)
    layers = [conv_2d, batch_norm]
    if use_activation: layers.append(activation)
    return nn.Sequential(*layers)


class XResNetBlock(nn.Module):
    """Creates the standard `XResNet` block."""
    def __init__(self, expansion: int, n_inputs: int, n_hidden: int,
                 coordconv: bool = False, stride: int = 1,
                 activation: torch.nn.Module = nn.ReLU(inplace=True)):
        super().__init__()

        n_inputs = n_inputs * expansion
        n_filters = n_hidden * expansion

        # convolution path
        if expansion == 1:
            layers = [conv_layer(n_inputs, n_hidden, 3, stride=stride, coordconv=coordconv),
                      conv_layer(n_hidden, n_filters, 3, zero_batch_norm=True, use_activation=False, coordconv=coordconv)]
        else:
            layers = [conv_layer(n_inputs, n_hidden, 1, coordconv=coordconv),
                      conv_layer(n_hidden, n_hidden, 3, stride=stride, coordconv=coordconv),
                      conv_layer(n_hidden, n_filters, 1, zero_batch_norm=True, use_activation=False, coordconv=coordconv)]

        self.convs = nn.Sequential(*layers)

        # identity path
        if n_inputs == n_filters:
            self.id_conv = nn.Identity()
        else:
            self.id_conv = conv_layer(n_inputs, n_filters, kernel_size=1, use_activation=False, coordconv=coordconv)
        if stride == 1:
            self.pool = nn.Identity()
        else:
            self.pool = nn.AvgPool2d(2, ceil_mode=True)

        self.activation = activation

    def forward(self, x):
        return self.activation(self.convs(x) + self.id_conv(self.pool(x)))


class XResNet(nn.Sequential):
    @classmethod
    def create(cls, expansion, layers, c_in=3, c_out=1000, coordconv=False):
        # create the stem of the network
        n_filters = [c_in, (c_in+1)*8, 64, 64]
        stem = [conv_layer(n_filters[i], n_filters[i+1], stride=2 if i==0 else 1, coordconv=coordconv)
                for i in range(3)]

        # create `XResNet` blocks
        n_filters = [64//expansion, 64, 128, 256, 512]

        res_layers = [cls._make_layer(expansion, n_filters[i], n_filters[i+1],
                                      n_blocks=l, coordconv=coordconv, stride=1 if i==0 else 2)
                      for i, l in enumerate(layers)]

        # putting it all together
        x_res_net = cls(*stem, nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                        *res_layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                        nn.Linear(n_filters[-1]*expansion, c_out)
                        )

        cls._init_module(x_res_net)
        return x_res_net

    @staticmethod
    def _make_layer(expansion, n_inputs, n_filters, n_blocks, stride, coordconv):
        return nn.Sequential(
            *[XResNetBlock(expansion, n_inputs if i==0 else n_filters, n_filters, coordconv, stride if i==0 else 1)
              for i in range(n_blocks)])

    @staticmethod
    def _init_module(module):
        if getattr(module, 'bias', None) is not None:
            nn.init.constant_(module.bias, 0)
        if isinstance(module, (nn.Conv2d,nn.Linear)):
            nn.init.kaiming_normal_(module.weight)
        # initialize recursively
        for l in module.children():
            XResNet._init_module(l)


def xresnet18(**kwargs): return XResNet.create(1, [2, 2,  2, 2], **kwargs)
def xresnet34(**kwargs): return XResNet.create(1, [3, 4,  6, 3], **kwargs)
def xresnet50(**kwargs): return XResNet.create(4, [3, 4,  6, 3], **kwargs)
def xresnet101(**kwargs): return XResNet.create(4, [3, 4, 23, 3], **kwargs)
def xresnet152(**kwargs): return XResNet.create(4, [3, 8, 36, 3], **kwargs)
