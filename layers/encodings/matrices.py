
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

from utils import *
from functions import *
from ..activations import *
from ..others import *

### ResNet

# class _Norm2d(nn.Sequential):
#     def __init__(self, n_channels):
#         super().__init__()
#         self.in_permute = Permute([0, 2, 3, 1])
#         self.norm = nn.LayerNorm(n_channels)
#         self.out_permute = Permute([0, -1, 1, 2])

class _Norm2d(nn.BatchNorm2d):
    pass

class _Activate(nn.LeakyReLU):
    def __init__(self):
        super().__init__(inplace=True)
#     pass

class DoubleConvPre(nn.Module):
    """([BN] => ReLU => convolution) * 2"""

    def __init__(self, in_channels, out_channels, dropout_rate=0.):
        super().__init__()
        self.double_conv = nn.Sequential(
            _Norm2d(out_channels),
            _Activate(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _Norm2d(out_channels),
            _Activate(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout(dropout_rate, inplace=False),
        )
        self.relu = _Activate()

    def forward(self, x, activate=False):
        if not activate:
            return self.double_conv(x)
        else:
            return self.relu(self.double_conv(x))

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, dropout_rate=0.):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _Norm2d(out_channels),
            _Activate(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _Norm2d(out_channels),
            nn.Dropout(dropout_rate, inplace=True),
        )
        self.relu = _Activate()

    def forward(self, x, activate=True):
        if not activate:
            return self.double_conv(x)
        else:
            return self.relu(self.double_conv(x))
    
class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.0, kernel_size=3):
        super().__init__()
        self.conv1 = self.make_conv(in_channels, out_channels, stride, kernel_size=kernel_size)
        self.bn1 = _Norm2d(out_channels)
        self.relu = _Activate()
        self.conv2 = self.make_conv(out_channels, out_channels, kernel_size=kernel_size)
        self.bn2 = _Norm2d(out_channels)
        self.downsample = downsample
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate, inplace=True)
            
    def make_conv(self, in_channels, out_channels, stride=1, kernel_size=3):
        if isinstance(kernel_size, int):
            padding = (kernel_size-1)//2
        elif isinstance(kernel_size, tuple):
            padding = ((kernel_size[0]-1)//2, (kernel_size[1]-1)//2)
        else:
            raise Exception(f'{kernel_size} ???')
            
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=False)

    def forward(self, x, activate=True):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.dropout_rate > 0:
            out = self.dropout(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        if activate:
            out = self.relu(out)
        return out
    

class ResNetBlock(nn.Sequential):
    
    def __init__(self, in_channels, out_channels, dropout_rate=0.):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        
        self.bn = _Norm2d(in_channels)
        self.res0 = ResConv(in_channels, in_channels, dropout_rate=dropout_rate)
        self.res1 = ResConv(in_channels, in_channels, dropout_rate=dropout_rate)
        self.res2 = ResConv(in_channels, out_channels, dropout_rate=dropout_rate)
        
        
class MultiKernelResNetBlock(nn.Module):
    
    def __init__(self, n_channels, dropout_rate=0., depth=3, kernel_list=[3,5,7]):
        super().__init__()
        
        self.n_channels = n_channels
        self.dropout_rate = dropout_rate
        
        self.depth = depth
        self.kernel_list = kernel_list
        
        self.bn = _Norm2d(n_channels)
        
        self.resKxK = nn.ModuleList([
            nn.ModuleList(
                [ResConv(n_channels, n_channels, dropout_rate=dropout_rate, 
                         kernel_size=kernel_size) for _ in range(self.depth)]
            ) for kernel_size in self.kernel_list
        ])
        
    def forward(self, x):
        
        x = self.bn(x)
        
        for i in range(self.depth):
            #x = (self.res3x3[i](x) + self.res5x5[i](x) + self.res7x7[i](x))
            x = sum(self.resKxK[j][i](x, activate=False) for j in range(len(self.kernel_list)))
            x = F.leaky_relu(x, inplace=True)
            
        return x / len(self.kernel_list)

    
## Custom Conv2d

class _CustomConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, *args, **kargs):
        super().__init__()
        
        self.args = args
        self.kargs = kargs
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
            
    def get_weight(self):
        return self.weight
        
    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, *self.args, **self.kargs)
        

class DiagConv2d(_CustomConv2d):
    def get_weight(self):
        weight = torch.zeros(
            [self.out_channels, self.in_channels, self.kernel_size, self.kernel_size], 
            dtype=self.weight.dtype, device=self.weight.device) # (Cout, Cin, K, K)
        weight.diagonal(dim1=-2, dim2=-1)[:] = self.weight
        return weight

class AntiDiagConv2d(_CustomConv2d):
    def get_weight(self):
        weight = torch.zeros(
            [self.out_channels, self.in_channels, self.kernel_size, self.kernel_size], 
            dtype=self.weight.dtype, device=self.weight.device) # (Cout, Cin, K, K)
        weight.diagonal(dim1=-2, dim2=-1)[:] = self.weight
        weight = weight.flip(-1)
        return weight
    
class DiagResConv(ResConv):
    def make_conv(self, in_channels, out_channels, stride=1, kernel_size=3):
        padding = (kernel_size-1)//2
            
        return DiagConv2d(in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=False)
class AntiDiagResConv(ResConv):
    def make_conv(self, in_channels, out_channels, stride=1, kernel_size=3):
        padding = (kernel_size-1)//2
            
        return DiagConv2d(in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=False)
    
class ACResNetBlock(nn.Module):
    
    def __init__(self, n_channels, dropout_rate=0., depth=3):
        super().__init__()
        
        self.n_channels = n_channels
        self.dropout_rate = dropout_rate
        
        self.depth = depth
        
        self.bn = _Norm2d(n_channels)
        
        self.res0_0 = ResConv(
            n_channels, n_channels, dropout_rate=dropout_rate, kernel_size=3)
        self.res1_0 = ResConv(
            n_channels, n_channels, dropout_rate=dropout_rate, kernel_size=3)
        self.res2_0 = ResConv(
            n_channels, n_channels, dropout_rate=dropout_rate, kernel_size=3)
        
        self.res0_1 = DiagResConv(
            n_channels, n_channels, dropout_rate=dropout_rate, kernel_size=3)
        self.res1_1 = DiagResConv(
            n_channels, n_channels, dropout_rate=dropout_rate, kernel_size=3)
        self.res2_1 = DiagResConv(
            n_channels, n_channels, dropout_rate=dropout_rate, kernel_size=3)
        
        self.res0_2 = AntiDiagResConv(
            n_channels, n_channels, dropout_rate=dropout_rate, kernel_size=3)
        self.res1_2 = AntiDiagResConv(
            n_channels, n_channels, dropout_rate=dropout_rate, kernel_size=3)
        self.res2_2 = AntiDiagResConv(
            n_channels, n_channels, dropout_rate=dropout_rate, kernel_size=3)
        
    def forward(self, x):
        
        x = self.bn(x)
        
        x = (self.res0_0(x, activate=False) + self.res0_1(x, activate=False) + self.res0_2(x, activate=False))
        x = F.leaky_relu(x, inplace=True)
        
        x = (self.res1_0(x, activate=False) + self.res1_1(x, activate=False) + self.res1_2(x, activate=False))
        x = F.leaky_relu(x, inplace=True)
        
        x = (self.res2_0(x, activate=False) + self.res2_1(x, activate=False) + self.res2_2(x, activate=False))
        x = F.leaky_relu(x, inplace=True)
        
        return x / 3
    
### UNet


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout_rate=0.):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetBlock(nn.Module):
    def __init__(self, n_channels, bilinear=True, dropout_rate=0.):
        super().__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.dropout_rate = dropout_rate
        
        self.inc = DoubleConv(n_channels, 128, dropout_rate=dropout_rate)
        
        self.down1 = Down(128, 256, dropout_rate=dropout_rate)
        self.down2 = Down(256, 512, dropout_rate=dropout_rate)
        self.down3 = Down(512, 512, dropout_rate=dropout_rate)
        self.up1 = Up(1024, 256, bilinear, dropout_rate=dropout_rate)
        self.up2 = Up(512, 128, bilinear, dropout_rate=dropout_rate)
        self.up3 = Up(256, n_channels, bilinear, dropout_rate=dropout_rate)

    def forward(self, x):
        if x.shape[2] >= 8:
            x1 = self.inc(x) # T
            x2 = self.down1(x1) # T//2 
            x3 = self.down2(x2) # T//4
            x4 = self.down3(x3) # T//8
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
        elif x.shape[2] >= 4:
            x1 = self.inc(x) # T
            x2 = self.down1(x1) # T//2 
            x3 = self.down2(x2) # T//4
            x = self.up2(x3, x2)
            x = self.up3(x, x1)
        else:
            x1 = self.inc(x) # T
            x2 = self.down1(x1) # T//2 
            x = self.up3(x2, x1)
        return x
            
class UNetPPBlock(nn.Module):
    def __init__(self, n_channels, bilinear=True, dropout_rate=0.):
        super().__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.dropout_rate = dropout_rate
        
        self.down = nn.MaxPool2d(2, 2)
        self._up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        n_hid_channels = 100
        
        self.inc = DoubleConv(n_channels, n_hid_channels, dropout_rate=dropout_rate)
        
        self.conv0_0 = DoubleConv(n_hid_channels, n_hid_channels, dropout_rate=dropout_rate)
        self.conv1_0 = DoubleConv(n_hid_channels, n_hid_channels, dropout_rate=dropout_rate)
        self.conv2_0 = DoubleConv(n_hid_channels, n_hid_channels, dropout_rate=dropout_rate)
        
        self.conv0_1 = DoubleConv(n_hid_channels, n_hid_channels, dropout_rate=dropout_rate)
        self.conv1_1 = DoubleConv(n_hid_channels, n_hid_channels, dropout_rate=dropout_rate)
        
        self.conv0_2 = DoubleConv(n_hid_channels, n_channels, dropout_rate=dropout_rate)
        
    def up(self, x1, x2=None):
        x1 = self._up(x1)
        if x2 is not None:
            # input is CHW
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        return x1
        
    def forward(self, x):
        
        if x.shape[2] >= 4:
            in0_0 = self.inc(x) # C T T
            out0_0 = self.conv0_0(in0_0) # C T T
            out1_0 = self.conv1_0(self.down(out0_0)) # C T/2 T/2
            out2_0 = self.conv2_0(self.down(out1_0)) # C T/4 T/4
            
#             print(out0_0.shape, out1_0.shape)
            out0_1 = self.conv0_1(out0_0 + self.up(out1_0, out0_0)) # C T T
#             print(out1_0.shape, out2_0.shape)
            out1_1 = self.conv1_1(out1_0 + self.up(out2_0, out1_0)) # C T/2 T/2
            
            out0_2 = self.conv0_2(out0_0 + out0_1 + self.up(out1_1, out0_0))
            
            x = out0_2
            
        else:
            
            in0_0 = self.inc(x) # C T T
            out0_0 = self.conv0_0(in0_0) # C T T
            out1_0 = self.conv1_0(self.down(out0_0)) # C T/2 T/2
            
            out0_1 = self.conv0_1(out0_0 + self.up(out1_0, out0_0)) # C T T
            
            x = out0_1
            
        return x
        
        
        
        
    