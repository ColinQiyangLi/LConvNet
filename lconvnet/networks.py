import torch
import torch.nn as nn

class Scaling(nn.Module):
    def __init__(self, scaling=1.0, learnable=False):
        super().__init__()
        self.scaling = nn.Parameter(
            torch.tensor(scaling).float(),
            requires_grad=learnable)

    def forward(self, x):
        return x * self.scaling


class MLP(nn.Module):
    def __init__(
            self, units, activation, linear_module, dropout_rate=0.0, scaling=None,
            dist_scaling=None):
        super().__init__()
        self.units = units
        self.activation = activation
        self.scaling = scaling
        self.linears = [
            linear_module(unit_in, unit_out) for unit_in,
                                                 unit_out in zip(self.units[: -1],
                                                                 self.units[1:])]
        self.dropouts = []
        for index, linear in enumerate(self.linears):
            self.add_module("linear{}".format(index), linear)
            self.dropouts.append(nn.Dropout(dropout_rate))
            if index != 0 and index != len(self.linears) - 1:
                self.add_module("dropout{}".format(index), self.dropouts[-1])
        if dist_scaling is not None:
            self.scaling_per_layer = dist_scaling ** (1.0 / (len(units) - 1))
            print(self.scaling_per_layer)
        else:
            self.scaling_per_layer = None

    def forward(self, x):
        for index, linear in enumerate(self.linears):
            if index != 0:
                x = self.activation(x)
            x = linear(x)
            if index != len(self.linears) - 1 and index != 0:
                x = self.dropouts[index](x)
            if self.scaling_per_layer is not None:
                x = self.scaling_per_layer * x
        if self.scaling is not None:
            x = x * self.scaling
        return x


class SmallConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, conv_module, linear_module,
                 activation, scaling, input_spatial_shape):
        super().__init__()
        self.conv1 = conv_module(in_channels=in_channels,
                                 out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv2 = conv_module(in_channels=16, out_channels=32,
                                 kernel_size=4, stride=2, padding=1)
        input_spatial_shape = tuple(map(lambda x: (x // 2) // 2, input_spatial_shape))
        self.linear1 = linear_module(
            in_features=input_spatial_shape[0] * input_spatial_shape[1] * 32,
            out_features=100)
        self.linear2 = linear_module(in_features=100, out_features=out_channels)
        self.activation = activation
        self.scaling_per_layer = scaling ** (1 / 4)

    def forward(self, x):
        x = self.conv1(x)
        x = x * self.scaling_per_layer
        x = self.activation(x)
        x = self.conv2(x)
        x = x * self.scaling_per_layer
        x = self.activation(x)
        x = self.linear1(x.flatten(start_dim=1))
        x = x * self.scaling_per_layer
        x = self.activation(x)
        x = self.linear2(x)
        x = x * self.scaling_per_layer
        return x

class LargeConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, conv_module, linear_module,
                 activation, scaling, input_spatial_shape, size_factor=1):
        super().__init__()
        self.conv1 = conv_module(in_channels=in_channels,
                                 out_channels=32 * size_factor, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv_module(in_channels=32 * size_factor, out_channels=32 * size_factor,
                                 kernel_size=4, stride=2, padding=1)
        self.conv3 = conv_module(in_channels=32 * size_factor,
                                 out_channels=64 * size_factor, kernel_size=3, stride=1, padding=1)
        self.conv4 = conv_module(in_channels=64 * size_factor, out_channels=64 * size_factor,
                                 kernel_size=4, stride=2, padding=1)
        input_spatial_shape = tuple(map(lambda x: (x // 2) // 2, input_spatial_shape))
        self.linear1 = linear_module(
            in_features=input_spatial_shape[0] * input_spatial_shape[1] * 64 * size_factor,
            out_features=512 * size_factor)
        self.linear2 = linear_module(in_features=512 * size_factor, out_features=512 * size_factor)
        self.linear3 = linear_module(in_features=512 * size_factor, out_features=out_channels)
        self.activation = activation
        self.scaling_per_layer = scaling ** (1 / 7)
        # import pdb; pdb.set_trace()

    def forward(self, x):
        x = self.conv1(x)
        x = x * self.scaling_per_layer
        x = self.activation(x)
        x = self.conv2(x)
        x = x * self.scaling_per_layer
        x = self.activation(x)
        x = self.conv3(x)
        x = x * self.scaling_per_layer
        x = self.activation(x)
        x = self.conv4(x)
        x = x * self.scaling_per_layer
        x = self.activation(x)
        x = self.linear1(x.flatten(start_dim=1))
        x = x * self.scaling_per_layer
        x = self.activation(x)
        x = self.linear2(x)
        x = x * self.scaling_per_layer
        x = self.activation(x)
        x = self.linear3(x)
        x = x * self.scaling_per_layer
        return x

class DCGANDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, conv_module, linear_module,
                 activation, scaling, input_spatial_shape, resize_to=(64, 64)):
        super().__init__()
        # Upsample to 64 x 64.
        self.upsample = nn.UpsamplingNearest2d(size=resize_to)

        # Assume input chn x 64 x 64
        self.conv1 = conv_module(in_channels=in_channels, out_channels=64,
                                 kernel_size=4, stride=2, padding=1)
        # 64 x 32 x 32
        self.conv2 = conv_module(in_channels=64, out_channels=128,
                                 kernel_size=4, stride=2, padding=1)
        # 128 x 16 x 16
        self.conv3 = conv_module(in_channels=128, out_channels=256,
                                 kernel_size=4, stride=2, padding=1)
        # 256 x 8 x 8
        self.conv4 = conv_module(in_channels=256, out_channels=512,
                                 kernel_size=4, stride=2, padding=1)
        # 512 x 4 x 4
        # self.conv5 = conv_module(in_channels=512 * 4 * 4, out_channels=out_channels,
        #                          kernel_size=1, stride=1, padding=0)
        self.linear = linear_module(in_features=512 * 4 * 4, out_features=1)
        # 1 * 1 * 1
        self.activation = activation
        self.scaling_per_layer = scaling ** (1 / 5)

    def forward(self, x):
        x = self.upsample(x)

        x = self.conv1(x)
        x = x * self.scaling_per_layer
        x = self.activation(x)

        x = self.conv2(x)
        x = x * self.scaling_per_layer
        x = self.activation(x)

        x = self.conv3(x)
        x = x * self.scaling_per_layer
        x = self.activation(x)

        x = self.conv4(x)
        x = x * self.scaling_per_layer
        x = self.activation(x)

        x = x.flatten(start_dim=1)
        x = self.linear(x)
        # x = x.view(-1, 512 * 4 * 4, 1, 1)
        # x = self.conv5(x)
        x = x * self.scaling_per_layer
        return x
