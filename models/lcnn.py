import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import numpy as np
from torch.nn import Parameter
from torch.autograd import Variable

sys.path.append("models")
from xlsr import SSLModel

# from .base_model import BaseModel


### A-softmax
def mypsi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)


class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)   # L2-norm along dimension 1, i.e. normalize each column
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos()) # size=(B,Classnum)
            k = (self.m*theta/3.14159265).floor() # theta >= k*pi / m => k <= m*theta/pi
            n_one = k*0.0 - 1
            psi_theta = (n_one**k) * cos_m_theta - 2*k    # size=(B,Classnum)
        else:
            theta = cos_theta.acos()
            psi_theta = mypsi(theta,self.m)
            psi_theta = psi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)   # ||x|| * cos_theta  
        psi_theta = psi_theta * xlen.view(-1,1)   # ||x|| * psi_theta
        output = (cos_theta, psi_theta)  # during evaluation, using cos_theta
        return output # size=(B,Classnum,2)

    def forward_eval(self, input):
        x = input
        w = self.weight

        ww = w.renorm(2,1,1e-5).mul(1e5)
        # xlen = x.pow(2).sum(1).pow(0.5)
        wlen = ww.pow(2).sum(0).pow(0.5)

        cos_theta = x.mm(ww) # size=(B,Classnum)
        # cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta / wlen.view(1,-1)     # ||x|| * cos_theta 
        # cos_theta = cos_theta.clamp(-1,1)
        # cos_theta = cos_theta * xlen.view(-1,1)   # ||x|| * cos_theta  

        return cos_theta


class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta, psi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1, target.data.view(-1,1), 1)
        index = index.byte()     #to uint8
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += psi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss
###

class mfm(nn.Module):
    """Max-Feature-Map."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1, dp_out=0.75):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Sequential(
                nn.Linear(in_channels, 2*out_channels),
                nn.Dropout(p=dp_out))

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x

class LCNN(nn.Module):
    def __init__(self, c_s=[128, 64, 32, 16, 8, 4, 2], asoftmax=True, phiflag=True, num_classes=2):
        super(LCNN, self).__init__()
        self.c_s = c_s
        self.asoftmax = asoftmax

        self.layer1 = nn.Sequential(
            mfm(1, c_s[5], 5, 1, 2),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False))  # 

        self.layer2 = nn.Sequential(
            group(c_s[5], c_s[4], 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False),  # 
            nn.BatchNorm2d(c_s[4]))

        self.layer3 = nn.Sequential(
            group(c_s[4], c_s[3], 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False),
            nn.BatchNorm2d(c_s[3]))  # 

        desired_width = 64
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, desired_width))  # x16 (c_s[3])
        
        self.fc0 = nn.Sequential(
            mfm(1024, 32, type=0, dp_out=0.75), 
            )

        self.fc1 = nn.Sequential(
            mfm(32, 32, type=0, dp_out=0.75), # 18 * 12 * 32
            )
        self.fc2 = nn.Sequential(
            mfm(32, 8, type=0, dp_out=0.0), # 18 * 12 * 32
            )


        if self.asoftmax:
            self.fc3 = AngleLinear(8, num_classes, phiflag=phiflag)
        else:
            self.fc3 = nn.Linear(8, num_classes)

        self.init_weight()

    def forward(self, x, eval=False):
        x = self.layer1(x)
        # print("layer 1 size: ", x.size())
        x = self.layer2(x)
        # print("layer 2 size: ", x.size())
        x = self.layer3(x)
        # print("layer 3 size: ", x.size())

        # Apply adaptive pooling
        x = self.adaptive_pool(x)
        
        # print("adaptive pool size: ", x.size(0))
        x = x.view(x.size(0), -1)
        # print("flatten size 0: ", x.size())
        x = self.fc0(x)
        # print("fc0 size: ", x.size())
        x = self.fc1(x)
        # print("fc1 size: ", x.size())
        x = self.fc2(x)
        if eval and self.asoftmax:
            x = self.fc3.forward_eval(x)
        else:
            x = self.fc3(x)
        # print("fc2 size: ", x.size())
        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(LCNN, self).__str__() + '\nTrainable parameters: {}'.format(params)

def lcnn_net(**kwargs):
    model = LCNN(**kwargs)
    return model


class ssl_lcnn(nn.Module):
    def __init__(self, device): 
        super(ssl_lcnn, self).__init__()
        self.frontend = SSLModel(device) 
        self.lcnn = lcnn_net(asoftmax=False).cuda()

    def forward(self, x):
        """combine the ssl and lcnn net
           output of `extract_feat` resides on GPU
           so we need to move se_resnet34 to GPU also, avoiding the
           <<RuntimeError: Input type (torch.cuda.FloatTensor) 
           and weight type (torch.FloatTensor) should be the same>>

        Args:
            x (Tensor): (batch_size, time series)
        """
        # print("x.shape before frontend", x.shape)
        x = self.frontend.extract_feat(x) # (batch_size, frame, 1024)
        # print("x.shape after frontend", x.shape)
        x = x.unsqueeze(1) # (batch_size, 1, frame, 1024)
        # print("x.shape after unsqueeze", x.shape)
        x = self.lcnn(x) # (batch_size, 2)
        # print("x.shape", x.shape)
        return x


if __name__ == '__main__':
    model = ssl_lcnn("cuda")
    audio_file = "/datac/longnv/audio_samples/ADD2023_T2_T_00000000.wav"
    import librosa
    import torch
    audio_data, _ = librosa.load(audio_file, sr=None)
    emb = model(torch.Tensor(audio_data).unsqueeze(0).to("cuda"))
    print(emb.shape)