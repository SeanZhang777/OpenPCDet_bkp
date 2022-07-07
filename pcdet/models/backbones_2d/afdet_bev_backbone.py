import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfCalibratedConv(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        half_channels = int(in_channels / 2)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, half_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(half_channels, eps=1e-3, momentum=0.01),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, half_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(half_channels, eps=1e-3, momentum=0.01),
                                   nn.ReLU())
        self.pooling = nn.AvgPool2d(kernel_size=4)
        self.conv3 = nn.Sequential(nn.Conv2d(half_channels, half_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(half_channels, eps=1e-3, momentum=0.01),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(half_channels, half_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(half_channels, eps=1e-3, momentum=0.01),
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(half_channels, half_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(half_channels, eps=1e-3, momentum=0.01),
                                   nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(half_channels, half_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(half_channels, eps=1e-3, momentum=0.01),
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.01),
                                   nn.ReLU())
    def forward(self, inputs):
        x = self.conv1(inputs)
        x2 = self.pooling(x)
        x2 = self.conv3(x2)
        x2 = F.interpolate(x2, scale_factor=(4, 4), mode='bilinear', align_corners=True)
        x1 = torch.sigmoid(x + x2)
        x3 = self.conv5(x)
        x = torch.mul(x1, x3)
        x = self.conv6(x)

        y = self.conv2(inputs)
        y = self.conv4(y)
        z = torch.concat([x, y], dim=1)
        z = self.conv7(z)
        output = z + inputs

        return output


class AFDetBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        output_conv_channel = self.model_cfg.get('OUTPUT_CONV_CHANNEL', 64)

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    SelfCalibratedConv(in_channels=num_filters[idx]),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
        if len(upsample_strides) > 0:
            stride = upsample_strides[0]
            if stride >= 1:
                self.deblocks.append(nn.Sequential(
                    nn.ConvTranspose2d(
                        num_filters[-1], num_upsample_filters[0],
                        upsample_strides[0],
                        stride=upsample_strides[0], bias=False
                    ),
                    nn.BatchNorm2d(num_upsample_filters[0], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ))
            else:
                stride = np.round(1 / stride).astype(np.int)
                self.deblocks.append(nn.Sequential(
                    nn.Conv2d(
                        num_filters[-1], num_upsample_filters[0],
                        stride,
                        stride=stride, bias=False
                    ),
                    nn.BatchNorm2d(num_upsample_filters[0], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ))

        self.conv1 = nn.Sequential(nn.Conv2d(num_filters[0], num_upsample_filters[0], kernel_size=1,
                                             stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(num_upsample_filters[0], eps=1e-3, momentum=0.01),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(num_upsample_filters[0] * 2, output_conv_channel, kernel_size=3,
                                             stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(output_conv_channel, eps=1e-3, momentum=0.01),
                                   nn.ReLU())
        self.num_bev_features = output_conv_channel

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']

        x = spatial_features
        assert len(self.blocks) == 2
        middle_feature = self.blocks[0](x)
        x = self.blocks[1](middle_feature)
        x = self.deblocks[0](x)
        y = self.conv1(middle_feature)
        z = torch.concat([x, y], dim=1)
        output = self.conv2(z)


        data_dict['spatial_features_2d'] = output

        return data_dict

if __name__ == '__main__':
    sc_conv = SelfCalibratedConv(128)
    input = torch.randn((2, 128, 512, 512))
    output = sc_conv(input)
    print(output.shape)