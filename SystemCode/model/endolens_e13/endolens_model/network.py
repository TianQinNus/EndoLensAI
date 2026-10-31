"""Portable supplied ResNet50/GN U-Net; all weights loaded from the release.

State keys and forward operations match the teammate's E13 architecture. No
ImageNet download or training-workspace import is performed by this module.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50

class DecoderBlock(nn.Module):
    def __init__(self,cin,skip_channels,cout):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(cin+skip_channels,cout,3,padding=1,bias=False),
            nn.GroupNorm(8,cout),nn.ReLU(inplace=True),
            nn.Conv2d(cout,cout,3,padding=1,bias=False),
            nn.GroupNorm(8,cout),nn.ReLU(inplace=True))
    def forward(self,x,skip=None,size=None):
        x=F.interpolate(x,size=skip.shape[-2:] if skip is not None else size,mode='bilinear',align_corners=False)
        if skip is not None:x=torch.cat((x,skip),dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=resnet50(weights=None);self.encoder.fc=nn.Identity()
        self.decoder=nn.ModuleList([DecoderBlock(2048,1024,256),DecoderBlock(256,512,128),
            DecoderBlock(128,256,64),DecoderBlock(64,64,32),DecoderBlock(32,0,16)])
        self.head=nn.Conv2d(16,1,1)
        self.register_buffer('input_mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('input_std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
    def forward(self,x):
        x=(x-self.input_mean)/self.input_std;size=x.shape[-2:];e=self.encoder
        f0=e.relu(e.bn1(e.conv1(x)));f1=e.layer1(e.maxpool(f0));f2=e.layer2(f1)
        f3=e.layer3(f2);f4=e.layer4(f3);z=f4
        for block,skip in zip(self.decoder[:4],(f3,f2,f1,f0)):z=block(z,skip)
        return self.head(self.decoder[4](z,size=size))
    def train(self,mode=True):
        super().train(mode)
        for module in self.encoder.modules():
            if isinstance(module,nn.BatchNorm2d):module.eval()
        return self
