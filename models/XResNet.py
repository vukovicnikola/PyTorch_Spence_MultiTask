import torch
import torch.nn as nn


def conv(n_inputs, n_filters, kernel_size=3, stride=1, bias=False) -> torch.nn.Conv2d:
    """Creates a convolution layer for `XResNet`."""
    return nn.Conv2d(n_inputs, n_filters,
                     kernel_size=kernel_size, stride=stride,
                     padding=kernel_size//2, bias=bias)


def conv_layer(n_inputs: int, n_filters: int,
               kernel_size: int = 3, stride=1,
               zero_batch_norm: bool = False, use_activation: bool = True,
               activation: torch.nn.Module = nn.ReLU(inplace=True)) -> torch.nn.Sequential:
    """Creates a convolution block for `XResNet`."""
    batch_norm = nn.BatchNorm2d(n_filters)
    # initialize batch normalization to 0 if its the final conv layer
    nn.init.constant_(batch_norm.weight, 0. if zero_batch_norm else 1.)
    conv_2d = conv(n_inputs, n_filters, kernel_size, stride=stride)
    layers = [conv_2d, batch_norm]
    if use_activation: layers.append(activation)
    return nn.Sequential(*layers)


class XResNetBlock(nn.Module):
    """Creates the standard `XResNet` block."""
    def __init__(self, expansion: int, n_inputs: int, n_hidden: int, stride: int = 1,
                 activation: torch.nn.Module = nn.ReLU(inplace=True)):
        super().__init__()

        n_inputs = n_inputs * expansion
        n_filters = n_hidden * expansion

        # convolution path
        if expansion == 1:
            layers = [conv_layer(n_inputs, n_hidden, 3, stride=stride),
                      conv_layer(n_hidden, n_filters, 3, zero_batch_norm=True, use_activation=False)]
        else:
            layers = [conv_layer(n_inputs, n_hidden, 1),
                      conv_layer(n_hidden, n_hidden, 3, stride=stride),
                      conv_layer(n_hidden, n_filters, 1, zero_batch_norm=True, use_activation=False)]

        self.convs = nn.Sequential(*layers)

        # identity path
        if n_inputs == n_filters:
            self.id_conv = nn.Identity()
        else:
            self.id_conv = conv_layer(n_inputs, n_filters, kernel_size=1, use_activation=False)
        if stride == 1:
            self.pool = nn.Identity()
        else:
            self.pool = nn.AvgPool2d(2, ceil_mode=True)

        self.activation = activation

    def forward(self, x):
        return self.activation(self.convs(x) + self.id_conv(self.pool(x)))


class XResNet(nn.Sequential):
    @classmethod
    def create(cls, expansion, layers, c_in=3, c_out=1000):
        # create the stem of the network
        n_filters = [c_in, (c_in+1)*8, 64, 64]
        stem = [conv_layer(n_filters[i], n_filters[i+1], stride=2 if i==0 else 1)
                for i in range(3)]

        # create `XResNet` blocks
        n_filters = [64//expansion, 64, 128, 256, 512]

        res_layers = [cls._make_layer(expansion, n_filters[i], n_filters[i+1],
                                      n_blocks=l, stride=1 if i==0 else 2)
                      for i, l in enumerate(layers)]

        # putting it all together
        x_res_net = cls(*stem, nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                        *res_layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                        nn.Linear(n_filters[-1]*expansion, c_out)
                        )

        cls._init_module(x_res_net)
        return x_res_net

    @staticmethod
    def _make_layer(expansion, n_inputs, n_filters, n_blocks, stride):
        return nn.Sequential(
            *[XResNetBlock(expansion, n_inputs if i==0 else n_filters, n_filters, stride if i==0 else 1)
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
