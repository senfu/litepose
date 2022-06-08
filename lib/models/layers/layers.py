import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class convbnrelu(nn.Sequential):
    def __init__(self, inp, oup, ker=3, stride=1, groups=1):
        super(convbnrelu, self).__init__(
            nn.Conv2d(inp, oup, ker, stride, ker // 2, groups=groups, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )

class Bottleneck(nn.Module):

    def __init__(self, inp, oup, s=1, k=3, r=4):
        super(Bottleneck, self).__init__()
        mid_dim = oup // r
        if inp == oup and s == 1:
            self.residual = True
        else:
            self.residual = False
        self.conv1 = nn.Conv2d(inp, mid_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_dim)
        self.conv2 = nn.Conv2d(mid_dim, mid_dim, kernel_size=k, stride=s, padding=k//2, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_dim)
        self.conv3 = nn.Conv2d(mid_dim, oup, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.residual == True:
            out += residual
        out = self.relu(out)
        return out

class UpConv(nn.Module):
    def __init__(self, inp, oup, k=3):
        super(UpConv, self).__init__()
        self.conv = nn.Conv2d(inp, oup, k, 1, k // 2, bias=False)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        x = self.conv(x)
        return x

class FusedMBConv(nn.Module):

    def __init__(self, inp, oup, s=1, k=3, r=4):
        super(FusedMBConv, self).__init__()
        feature_dim = _make_divisible(round(inp * r), 8)
        self.inv = nn.Sequential(
            nn.Conv2d(inp, feature_dim, k, s, k // 2, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU6(inplace = True)
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(feature_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )
        self.use_residual_connection = s == 1 and inp == oup
        
    def forward(self, x):
        out = self.inv(x)
        out = self.point_conv(out)
        if self.use_residual_connection:
            out += x
        return out

class InvBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, ker=3, exp=6):
        super(InvBottleneck, self).__init__()
        feature_dim = _make_divisible(round(inplanes * exp), 8)
        self.inv = nn.Sequential(
            nn.Conv2d(inplanes, feature_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU6(inplace = True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, ker, stride, ker // 2, groups=feature_dim, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU6(inplace = True)
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(feature_dim, planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.stride = stride
        self.use_residual_connection = stride == 1 and inplanes == planes
        
    def forward(self, x):
        out = self.inv(x)
        out = self.depth_conv(out)
        out = self.point_conv(out)
        if self.use_residual_connection:
            out += x
        return out

class SepConv2d(nn.Module):
    def __init__(self, inp, oup, ker=3, stride=1):
        super(SepConv2d, self).__init__()
        conv = [
            nn.Conv2d(inp, inp, ker, stride, ker // 2, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        ]
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        output = self.conv(x)
        return output




# from https://github.com/tristandb/EfficientDet-PyTorch/blob/b86f3661c9167ed9394bdfd430ea4673ad5177c7/bifpn.py
class DepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution. 
    
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(DepthwiseConvBlock,self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                               padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()
        
    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)
    
class ConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.act(x)

class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """
    def __init__(self, feature_size=64, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        
        self.p2_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p3_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_td = DepthwiseConvBlock(feature_size, feature_size)
        
        self.p3_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p7_out = DepthwiseConvBlock(feature_size, feature_size)
        
        # TODO: Init weights
        self.w1 = nn.Parameter(torch.Tensor(2, 5))
        self.w1_relu = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, 5))
        self.w2_relu = nn.ReLU()
    
    def forward(self, inputs):
        p2_x, p3_x, p4_x, p5_x, p6_x, p7_x = inputs
        
        # Calculate Top-Down Pathway
        w1 = self.w1_relu(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.epsilon
        w2 = self.w2_relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.epsilon
        
        p7_td = p7_x
        p6_td = self.p6_td(w1[0, 0] * p6_x + w1[1, 0] * F.interpolate(p7_td, scale_factor=2))        
        p5_td = self.p5_td(w1[0, 1] * p5_x + w1[1, 1] * F.interpolate(p6_td, scale_factor=2))
        p4_td = self.p4_td(w1[0, 2] * p4_x + w1[1, 2] * F.interpolate(p5_td, scale_factor=2))
        p3_td = self.p3_td(w1[0, 3] * p3_x + w1[1, 3] * F.interpolate(p4_td, scale_factor=2))
        p2_td = self.p2_td(w1[0, 4] * p2_x + w1[1, 4] * F.interpolate(p3_td, scale_factor=2))
        
        # Calculate Bottom-Up Pathway
        p2_out = p2_td
        p3_out = self.p3_out(w2[0, 0] * p3_x + w2[1, 0] * p3_td + w2[2, 0] * nn.Upsample(scale_factor=0.5)(p2_out))
        p4_out = self.p4_out(w2[0, 1] * p4_x + w2[1, 1] * p4_td + w2[2, 1] * nn.Upsample(scale_factor=0.5)(p3_out))
        p5_out = self.p5_out(w2[0, 2] * p5_x + w2[1, 2] * p5_td + w2[2, 2] * nn.Upsample(scale_factor=0.5)(p4_out))
        p6_out = self.p6_out(w2[0, 3] * p6_x + w2[1, 3] * p6_td + w2[2, 3] * nn.Upsample(scale_factor=0.5)(p5_out))
        p7_out = self.p7_out(w2[0, 4] * p7_x + w2[1, 4] * p7_td + w2[2, 4] * nn.Upsample(scale_factor=0.5)(p6_out))

        return [p2_out, p3_out, p4_out, p5_out, p6_out, p7_out]
    
class BiFPN(nn.Module):
    def __init__(self, size, feature_size=64, num_layers=2, epsilon=0.0001):
        super(BiFPN, self).__init__()
        self.p2 = nn.Conv2d(size[0], feature_size, kernel_size=1, stride=1, padding=0)
        self.p3 = nn.Conv2d(size[1], feature_size, kernel_size=1, stride=1, padding=0)
        self.p4 = nn.Conv2d(size[2], feature_size, kernel_size=1, stride=1, padding=0)
        self.p5 = nn.Conv2d(size[3], feature_size, kernel_size=1, stride=1, padding=0)
        
        # p6 is obtained via a 3x3 stride-2 conv on C5
        self.p6 = nn.Conv2d(size[3], feature_size, kernel_size=3, stride=2, padding=1)
        
        # p7 is computed by applying ReLU followed by a 3x3 stride-2 conv on p6
        self.p7 = ConvBlock(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNBlock(feature_size))
        self.bifpn = nn.Sequential(*bifpns)
    
    def forward(self, inputs):
        c2, c3, c4, c5 = inputs
        
        # Calculate the input column of BiFPN
        p2_x = self.p2(c2)
        p3_x = self.p3(c3)        
        p4_x = self.p4(c4)
        p5_x = self.p5(c5)
        p6_x = self.p6(c5)
        p7_x = self.p7(p6_x)

        features = [p2_x, p3_x, p4_x, p5_x, p6_x, p7_x]
        return self.bifpn(features)
