import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

def SnConv1(in_channel, out_channel, bias = True):
        return spectral_norm(nn.Conv2d(in_channel, out_channel, 1, 1, bias = bias))

def SnConv3(in_channel, out_channel, bias = True):
        return spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias = bias))

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_channels, which_conv = SnConv1):
        super(Self_Attn, self).__init__()
        self.ch = in_channels
        self.which_conv = which_conv
        self.theta = self.which_conv(self.ch, self.ch // 8, bias=False)
        self.phi = self.which_conv(self.ch, self.ch // 8, bias=False)
        self.g = self.which_conv(self.ch, self.ch // 2, bias=False)
        self.o = self.which_conv(self.ch // 2, self.ch, bias=False)
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])    
        # Perform reshapes
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x

class ConditionalNorm2d(nn.Module):
    def __init__(self, channels, num_features, norm_type="bn"):
        super(ConditionalNorm2d, self).__init__()
        self.channels = channels
        if norm_type == "bn":
            self.norm = nn.BatchNorm2d(channels, affine=False)
        elif norm_type == "in":
            self.norm = nn.InstanceNorm2d(channels, affine=False)
        else:
            raise ValueError("Normalisation type not recognised.")

        self.fcw = nn.Linear(num_features, channels)
        self.fcb = nn.Linear(num_features, channels)

    def forward(self, x, features):
        out = self.norm(x)
        w   = self.fcw(features)
        b   = self.fcb(features)

        out =  w.view(-1, self.channels, 1, 1) * out + b.view(-1, self.channels, 1, 1)
        return out
    
class Res_down(nn.Module):
    def __init__(self, in_channel, out_channel, down_sample = True, first_block = False):
        super(Res_down, self).__init__()
        
        self.conv1  = SnConv3(in_channel, in_channel)
        self.conv2  = SnConv3(in_channel, out_channel)
        self.conv3  = SnConv1(in_channel, out_channel)

        self.AvePool1 = nn.AvgPool2d(2, 2)
        self.AvePool2 = nn.AvgPool2d(2, 2)
        
        self.down_sample = down_sample
        self.first_block = first_block

    def forward(self, x):
        skip = self.conv3(x)
        
        if not self.first_block:
            x = F.relu(x)
            
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        
        if self.down_sample:
            skip = self.AvePool2(skip)
            x = self.AvePool1(x)

        output = x + skip
        return output

class Res_up(nn.Module):
    def __init__(self, in_channel, out_channel, num_features, norm_type="bn"):
        super(Res_up, self).__init__()
        
        self.conv1  = SnConv3(in_channel, in_channel)
        self.conv2  = SnConv3(in_channel, out_channel)
        self.conv3  = SnConv1(in_channel, out_channel)
        
        self.CN1    = ConditionalNorm2d(in_channel, num_features, norm_type=norm_type)
        self.CN2    = ConditionalNorm2d(in_channel, num_features, norm_type=norm_type)

        self.UpNN1 = nn.Upsample(scale_factor = 2, mode='nearest')
        self.UpNN2 = nn.Upsample(scale_factor = 2, mode='nearest')
        
    def forward(self, x, features):
        skip = self.conv3(self.UpNN2(x))
        
        x = F.relu(self.CN1(x, features))
        x = self.UpNN1(x)
        x = F.relu(self.CN2(self.conv1(x), features))
        x = self.conv2(x)
        
        output = x + skip
        return output

    
class Generator256(nn.Module):
    def __init__(self, in_noise, in_features, ch = 32, norm_type="bn"):
        super(Generator256, self).__init__()
        self.linear1  = spectral_norm(nn.Linear(in_noise, ch*16*4*4))
        self.ResUp1 = Res_up(ch*16, ch*16, in_features, norm_type=norm_type)
        self.ResUp2 = Res_up(ch*16, ch*8, in_features, norm_type=norm_type)
        self.ResUp3 = Res_up(ch*8,  ch*8, in_features, norm_type=norm_type)
        self.ResUp4 = Res_up(ch*8,  ch*4, in_features, norm_type=norm_type)
        self.attention = Self_Attn(ch*4)
        self.ResUp5 = Res_up(ch*4,  ch*2, in_features, norm_type=norm_type)
        self.ResUp6 = Res_up(ch*2,  ch, in_features, norm_type=norm_type)

        if norm_type == "bn":
            self.Norm1  = nn.BatchNorm2d(ch, momentum=0.0001)
        elif norm_type == "in":
            self.Norm1 = nn.InstanceNorm2d(ch)
        else:
            raise ValueError("Normalisation type not recognised.")
            
        self.Conv2  = SnConv3(ch, 3)         
        self.Tanh = nn.Tanh()

    def forward(self, x, features):
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = x.view(x.shape[0], -1, 4, 4)#4x4
        
        x = self.ResUp1(x, features)#8x8
        x = self.ResUp2(x, features)#16x16
        x = self.ResUp3(x, features)#32x32
        x = self.ResUp4(x, features)#64x64
        x = self.attention(x)       #64x64
        x = self.ResUp5(x, features)#128x128
        x = self.ResUp6(x, features)#256x256
        x = F.relu(self.Norm1(x))
        x = self.Conv2(x)
        return self.Tanh(x)
                   
class Discriminator256(nn.Module):
    def __init__(self, channels, in_features, ch = 32, norm_type="bn"):
        super(Discriminator256, self).__init__()
        self.Conv1 = SnConv3(channels, ch)
        self.Res_down1 = Res_down(ch, ch*2)
        self.Res_down2 = Res_down(ch*2, ch*4)
        self.attention = Self_Attn(ch*4)
        self.Res_down3 = Res_down(ch*4,  ch*8)
        self.Res_down4 = Res_down(ch*8,  ch*8)
        self.Res_down5 = Res_down(ch*8,  ch*16)
        self.Res_down6 = Res_down(ch*16, ch*16)
        self.CN1    = ConditionalNorm2d(ch*16, in_features, norm_type=norm_type)
        self.Conv2  = spectral_norm(nn.Conv2d(ch*16, 1, kernel_size = 4, stride=1))

    def forward(self, x, features):
        x = self.Conv1(x)
        x = self.Res_down1(x)#128x128
        x = self.Res_down2(x)#64x64
        x = self.attention(x)#64x64
        x = self.Res_down3(x)#32x32
        x = self.Res_down4(x)#16x16
        x = self.Res_down5(x)#8x8
        x = self.Res_down6(x)#4x4
        x = F.relu(self.CN1(x, features))
        x = self.Conv2(x)#1x1
        return x
         
