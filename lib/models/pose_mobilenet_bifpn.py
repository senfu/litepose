# TODO: DEBUG ONLY!!!
import sys

from matplotlib.pyplot import xlim
sys.path.append("/home/junyan/litepose")
sys.path.append("/home/junyan/litepose/lib")

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from lib.models.layers.layers import InvBottleneck, convbnrelu, SepConv2d, BiFPN

def rand(c):
    return random.randint(0, c - 1)

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class LitePoseBiFPN(nn.Module):
    def __init__(self, cfg, width_mult=1.0, round_nearest=8, cfg_arch=None):
        super(LitePoseBiFPN, self).__init__()
        backbone_setting = cfg_arch['backbone_setting']
        input_channel = cfg_arch['input_channel']
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.first = nn.Sequential(
            convbnrelu(3, 32, ker=3, stride=2),
            convbnrelu(32, 32, ker=3, stride=1, groups=32),
            nn.Conv2d(32, input_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(input_channel)
        )
        self.channel = [input_channel]
        # building inverted residual blocks
        self.stage = []
        for id_stage in range(len(backbone_setting)):
            n = backbone_setting[id_stage]['num_blocks']
            s = backbone_setting[id_stage]['stride']
            c = backbone_setting[id_stage]['channel']
            c = _make_divisible(c * width_mult, round_nearest)
            block_setting = backbone_setting[id_stage]['block_setting']
            layer = []
            for id_block in range(n):
                t, k = block_setting[id_block]
                stride = s if id_block == 0 else 1
                layer.append(InvBottleneck(input_channel, c, stride, ker=k, exp=t))
                input_channel = c
            layer = nn.Sequential(*layer)
            self.stage.append(layer)
            self.channel.append(c)
        self.stage = nn.ModuleList(self.stage)
        self.bifpn = BiFPN(self.channel[-4:], feature_size=self.channel[-1], num_layers=3)
        self.out2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            SepConv2d(self.channel[-1], self.channel[-1], 5),
            nn.Conv2d(self.channel[-1], cfg.MODEL.NUM_JOINTS, 1)
        )
        self.out1 = nn.Sequential(
            SepConv2d(self.channel[-1], self.channel[-1], 5),
            nn.Conv2d(self.channel[-1], cfg.MODEL.NUM_JOINTS * 2, 1)
        )

    def forward(self, x):
        x = self.first(x)
        x_list = [x]
        for i in range(len(self.stage)):
            tmp = self.stage[i](x_list[-1])
            x_list.append(tmp)
        x_list = self.bifpn(x_list[-4:])
        out1 = self.out1(x_list[0])
        out2 = self.out2(x_list[0])
        final_outputs = [out1, out2]
        return final_outputs


def get_pose_net(cfg, is_train=False, cfg_arch=None):
    model = LitePoseBiFPN(cfg, cfg_arch=cfg_arch)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        print(cfg.MODEL.PRETRAINED)
        if os.path.isfile(cfg.MODEL.PRETRAINED):
            print("load pre-train model")
            need_init_state_dict = {}
            state_dict = torch.load(cfg.MODEL.PRETRAINED, map_location=torch.device('cpu'))
            for key, value in state_dict.items():
                need_init_state_dict[key] = value
            try:
                model.load_state_dict(need_init_state_dict, strict=False)
            except:
                print("Error load!")
    return model


if __name__ == "__main__":
    from lib.config import cfg, update_config
    import json
    from arch_manager import ArchManager
    from ptflops import get_model_complexity_info
    import torch
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser(description='Train keypoints network')
        # general
        parser.add_argument('--cfg',
                            help='experiment configure file name',
                            required=True,
                            type=str)

        parser.add_argument('opts',
                            help="Modify config options using the command-line",
                            default=None,
                            nargs=argparse.REMAINDER)

        # distributed training
        parser.add_argument('--gpu',
                            help='gpu id for multiprocessing training',
                            type=str)
        parser.add_argument('--world-size',
                            default=1,
                            type=int,
                            help='number of nodes for distributed training')
        parser.add_argument('--dist-url',
                            default='tcp://127.0.0.1:23459',
                            type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--rank',
                            default=0,
                            type=int,
                            help='node rank for distributed training')
        # fixed config for supernet
        parser.add_argument('--superconfig',
                            default=None,
                            type=str,
                            help='fixed arch for supernet training')
        # distillation
        parser.add_argument('--teacher',
                            default=False,
                            type=str,
                            help='teacher model path for distillation')

        args = parser.parse_args()

        return args
    args = parse_args()
    update_config(cfg, args)
    fixed_arch = None
    if args.superconfig is not None:
        with open(args.superconfig, 'r') as f:
           fixed_arch = json.load(f)
    arch_manager = ArchManager(cfg)
    cfg_arch = arch_manager.fixed_sample()
    if fixed_arch is not None:
        cfg_arch = fixed_arch
    model = LitePoseBiFPN(cfg, cfg_arch=cfg_arch)
    macs, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat=False, as_strings=True, verbose=True)
    print("MACs:", macs)
    print("Params:", params)