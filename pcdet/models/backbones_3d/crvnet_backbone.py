import torch
import torch.nn as nn
import numpy as np

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        # self.nonlinearity = nn.LeakyReLU(0.1)
        self.nonlinearity = nn.LeakyReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = self.conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = self.conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)

    def conv_bn(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        result = nn.Sequential()
        result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
        result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
        return result

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

class CSPRepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride=1, split_ratio=0.5, deploy=False):
        super(CSPRepBlock, self).__init__()
        self.deploy=deploy
        self.split_channels_left = int(out_channels * split_ratio)
        self.split_channels_right = out_channels - self.split_channels_left

        self.block_first_conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

        self.block_left_conv1x1 = nn.Sequential(
            nn.Conv2d(out_channels, self.split_channels_left, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.split_channels_left),
            nn.LeakyReLU()
        )
        self.block_left  = nn.Sequential(
            *[RepVGGBlock(self.split_channels_left, self.split_channels_left, kernel_size=3, padding=1, deploy=self.deploy)
                for i in range(num_blocks)]
        )

        self.block_right = nn.Sequential(
            nn.Conv2d(out_channels, self.split_channels_right, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.split_channels_right),
            nn.LeakyReLU()
        )

        self.block_last_conv1x1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.block_first_conv3x3(x)

        left  = self.block_left_conv1x1(x)
        left  = self.block_left(left)
        right = self.block_right(x)

        output = torch.cat((left, right), dim=1)
        output = self.block_last_conv1x1(output)
        return output

class CRVNetBackbone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.deploy = model_cfg.get('DEPLOY', False)
        self.blocks_num = [6, 16, 1, 1]
        self.sparse_shape = grid_size[[1, 0]]

        self.conv1 = RepVGGBlock(input_channels, 32, kernel_size=3, stride=1, padding=1, deploy=self.deploy)
        self.conv2 = CSPRepBlock(32, 64, stride=2, num_blocks=6, deploy=self.deploy)
        self.conv3 = CSPRepBlock(64, 128, stride=2, num_blocks=16, deploy=self.deploy)
        self.conv4 = CSPRepBlock(128, 256, stride=2, num_blocks=1, deploy=self.deploy)
        self.conv5 = CSPRepBlock(256, 256, stride=2, num_blocks=1, deploy=self.deploy)

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }

    def forward(self, batch_dict):
        spatial_features = batch_dict['spatial_features']
        batch_size = batch_dict['batch_size']

        x_conv1 = self.conv1(spatial_features)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)

        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })

        return batch_dict