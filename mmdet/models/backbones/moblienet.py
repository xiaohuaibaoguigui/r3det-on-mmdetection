import torch.nn as nn
from ..builder import BACKBONES

from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger

import math
from torchtoolbox.nn import Activation
from functools import partial
import torch.nn.functional as F


def make_divisible(x, divisible_by=8):
    return int(math.ceil(x * 1. / divisible_by) * divisible_by)

class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.
    
class SE_Module(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SE_Module, self).__init__()
        reduction_c = make_divisible(channels // reduction)
        self.out = nn.Sequential(
            nn.Conv2d(channels, reduction_c, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_c, channels, 1, bias=True),
            HardSigmoid()
        )

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1)
        y = self.out(y)
        return x * y
    
class MobileNetBottleneck(nn.Module):
    def __init__(self, in_c, expansion, out_c, kernel_size, stride, se=False,
                 activation='relu6', first_conv=True, skip=True, linear=True):
        super(MobileNetBottleneck, self).__init__()

        self.act = Activation(activation, auto_optimize=True)  # [bug]no use when linear=True
        hidden_c = round(in_c * expansion)
        self.linear = linear
        self.skip = stride == 1 and in_c == out_c and skip

        seq = []
        if first_conv and in_c != hidden_c:
            seq.append(nn.Conv2d(in_c, hidden_c, 1, 1, bias=False))
            seq.append(nn.BatchNorm2d(hidden_c))
            seq.append(Activation(activation, auto_optimize=True))
            
        seq.append(nn.Conv2d(hidden_c, hidden_c, kernel_size, stride,
                             kernel_size // 2, groups=hidden_c, bias=False))
        seq.append(nn.BatchNorm2d(hidden_c))
        seq.append(Activation(activation, auto_optimize=True))
        if se:
            seq.append(SE_Module(hidden_c))
        seq.append(nn.Conv2d(hidden_c, out_c, 1, 1, bias=False))
        seq.append(nn.BatchNorm2d(out_c))

        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        skip = x
        x = self.seq(x)
        if self.skip:
            x = skip + x
        if not self.linear:
            x = self.act(x)
        return x

@BACKBONES.register_module()
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000, small_input=False):
        super(MobileNetV1, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2 if not small_input else 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        MB1_Bottleneck = partial(MobileNetBottleneck, first_conv=False,
                                 activation='relu', skip=False, linear=False)
        # 400
#         self.mb_block = nn.Sequential(
#             MB1_Bottleneck(32, 1, 64, 3, 1),

#             MB1_Bottleneck(64, 1, 128, 3, 2), 
#             MB1_Bottleneck(128, 1, 128, 3, 1),

#             MB1_Bottleneck(128, 1, 256, 3, 2),
#             MB1_Bottleneck(256, 1, 256, 3, 1),

#             MB1_Bottleneck(256, 1, 512, 3, 2), 
#             MB1_Bottleneck(512, 1, 512, 3, 1),
#             MB1_Bottleneck(512, 1, 512, 3, 1),
#             MB1_Bottleneck(512, 1, 512, 3, 1),
#             MB1_Bottleneck(512, 1, 512, 3, 1),
#             MB1_Bottleneck(512, 1, 512, 3, 1),

#             MB1_Bottleneck(512, 1, 1024, 3, 2), 
#             MB1_Bottleneck(1024, 1, 1024, 3, 1),
#         )
        self.stage_start = nn.Sequential(MB1_Bottleneck(32, 1, 64, 3, 1)) # 400 -> 400
        
        self.stage0 = nn.Sequential(MB1_Bottleneck(64, 1, 128, 3, 2),
                                    MB1_Bottleneck(128, 1, 128, 3, 1)) # 400 -> 200
        
        self.stage1 = nn.Sequential(MB1_Bottleneck(128, 1, 256, 3, 2), 
                                    MB1_Bottleneck(256, 1, 256, 3, 1)) # 200 -> 100
        
        self.stage2 = nn.Sequential(MB1_Bottleneck(256, 1, 512, 3, 2), 
                                    MB1_Bottleneck(512, 1, 512, 3, 1),
                                    MB1_Bottleneck(512, 1, 512, 3, 1),
                                    MB1_Bottleneck(512, 1, 512, 3, 1),
                                    MB1_Bottleneck(512, 1, 512, 3, 1),
                                    MB1_Bottleneck(512, 1, 512, 3, 1)) # 100 -> 50
        
        self.stage3 = nn.Sequential(MB1_Bottleneck(512, 1, 1024, 3, 2), 
                                    MB1_Bottleneck(1024, 1, 1024, 3, 1)) # 50 -> 25
#         self.last_block = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#         )
#         self.output = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.first_block(x)
        x = self.stage_start(x)
        outs = []
        x = self.stage0(x)
        outs.append(x) # 200, 128
        x = self.stage1(x)
        outs.append(x) # 100, 256
        x = self.stage2(x)
        outs.append(x) # 50,  512
        x = self.stage3(x)
        outs.append(x) # 25,  1024
#         x = self.mb_block(x)    
#         x = self.last_block(x)
#         x = self.output(x)
        return tuple(outs)

    # dummy function
    def init_weights(self, pretrained=None):
        if pretrained is None:
            print("[MobileNetV1]Train MobileNet V1 from scratch...")
        else:
            if isinstance(pretrained, str):
                print("[MobileNetV1]Train MobileNet V1 from weights:", pretrained)
                logger = get_root_logger()
                load_checkpoint(self, pretrained, strict=False, logger=logger)
            else:
                raise TypeError('pretrained must be a str or None')
        