import torch
import torch.nn as nn
from torch.nn import functional as F 

class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        return out

class BottleNeck(nn.Module):
    def __init__(self,in_channels,out_channels,stride,cardinality):
        super(BottleNeck,self).__init__()
        self.conv_reduce=nn.Conv2d(in_channels,int(in_channels//2),kernel_size=1,padding=0,stride=1,bias=False)
        self.bn1=nn.BatchNorm2d(int(in_channels//2))

        self.conv=nn.Conv2d(int(in_channels//2),int(in_channels//2),kernel_size=3,padding=1,stride=stride,groups=cardinality,bias=False)
        self.bn2=nn.BatchNorm2d(int(in_channels//2))

        self.conv_expand=nn.Conv2d(int(in_channels//2),out_channels,kernel_size=1,padding=0,stride=1,bias=False)

        self.relu=nn.ReLU(inplace=True)
        self.isequal=(in_channels==out_channels)
        self.shortcut=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0,stride=stride,bias=False),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self,x):
        out=self.relu(self.bn1(self.conv_reduce(x)))
        out=self.relu(self.bn2(self.conv(out)))
        out=self.conv_expand(out)
        if self.isequal==1:
            return x+out
        else:
            return self.shortcut(x)+out
        

#Resnet 18 PAN
class Student1(nn.Module):
    def __init__(self):
        super(Student1,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(64)
        )
        self.blk1_1=ResBlk(64,128,2)
        self.blk1_2=ResBlk(128,128,1)
        
        self.blk2_1=ResBlk(128,256,2)
        self.blk2_2=ResBlk(256,256,1)

        self.blk3_1=ResBlk(256,512,2)
        self.blk3_2=ResBlk(512,512,1)

        self.blk4_1=ResBlk(512,1024,2)
        self.blk4_2=ResBlk(1024,1024,1)

        self.FC=nn.Linear(1024,10)
    
    def forward(self,x):
        out=self.conv1(x)
        out=self.blk1_1(out)
        out=self.blk1_2(out)

        out=self.blk2_1(out)
        out=self.blk2_2(out)

        out=self.blk3_1(out)
        out=self.blk3_2(out)

        out=self.blk4_1(out)
        out=self.blk4_2(out)

        out=F.adaptive_avg_pool2d(out,[1,1])
        out=out.view(out.size()[0],-1)
        out=self.FC(out)
        
        return out

class Student1_upgraded_version1(nn.Module):
    def __init__(self):
        super(Student1_upgraded_version1,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(32)
        )
        self.blk1_1=BottleNeck(32,64,2,16)
        self.blk1_2=BottleNeck(64,64,1,32)

        
        self.blk2_1=BottleNeck(64,128,2,32)
        self.blk2_2=BottleNeck(128,128,1,32)

        self.blk3_1=BottleNeck(128,256,2,64)
        self.blk3_2=BottleNeck(256,256,1,64)

        self.blk4_1=BottleNeck(256,512,2,64)
        self.blk4_2=BottleNeck(512,512,1,64)

        self.blk5_1=BottleNeck(512,1024,2,64)
        self.blk5_2=BottleNeck(1024,1024,1,64)

        self.FC=nn.Linear(1024,10)
    
    def forward(self,x):
        out=self.conv1(x)
        out=self.blk1_1(out)
        out=self.blk1_2(out)

        out=self.blk2_1(out)
        out=self.blk2_2(out)

        out=self.blk3_1(out)
        out=self.blk3_2(out)

        out=self.blk4_1(out)
        out=self.blk4_2(out)

        out=self.blk5_1(out)
        out=self.blk5_2(out)

        out=F.adaptive_avg_pool2d(out,[1,1])
        out=out.view(out.size()[0],-1)
        out=self.FC(out)
        
        return out

#ResNeXt 26 MS
class Student2(nn.Module):
    def __init__(self):
        super(Student2,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(4,64,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(64)
        )
        self.blk1_1=BottleNeck(64,128,2,32)
        self.blk1_2=BottleNeck(128,128,1,32)
        
        self.blk2_1=BottleNeck(128,256,2,32)
        self.blk2_2=BottleNeck(256,256,1,32)

        self.blk3_1=BottleNeck(256,512,2,64)
        self.blk3_2=BottleNeck(512,512,1,64)

        self.blk4_1=BottleNeck(512,1024,2,64)
        self.blk4_2=BottleNeck(1024,1024,1,64)

        self.FC=nn.Linear(1024,10)
    
    def forward(self,x):
        out=self.conv1(x)
        out=self.blk1_1(out)
        out=self.blk1_2(out)

        out=self.blk2_1(out)
        out=self.blk2_2(out)

        out=self.blk3_1(out)
        out=self.blk3_2(out)

        out=self.blk4_1(out)
        out=self.blk4_2(out)

        out=F.adaptive_avg_pool2d(out,[1,1])
        out=out.view(out.size()[0],-1)
        out=self.FC(out)
        
        return out

def IHS(ms,pan):
    batch_size=ms.size()[0]
    h_pan,w_pan=pan.size()[2],pan.size()[3]
    ms_upsample=F.interpolate(ms,size=(h_pan,w_pan),mode='bilinear',align_corners=False)
    chanenls_ms=ms.size()[1]
    I=(1/3)*(ms_upsample[:,0,:,:]+0.75*ms_upsample[:,1,:,:]+0.25*ms_upsample[:,2,:,:]+ms_upsample[:,3,:,:])
    g_ms=0.25
    # print("I:",I.size())
    # print("PAN:",pan[:,0,:,:].size())
    fusion_factor=g_ms*(pan[:,0,:,:]-I)
    # print("fusion_factor:",fusion_factor.size())
    ms_fusion=ms_upsample+fusion_factor.reshape(batch_size,1,h_pan,w_pan)
    # print("ms_fusion:",ms_fusion.size())
    return ms_fusion

#ResNeXt 60 fusion_data
class Teacher(nn.Module):
    def __init__(self):
        super(Teacher,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(4,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64)
        )
        groups1=[]
        groups2=[]
        groups3=[]
        groups4=[]
        for i in range(5):
            groups1.append(BottleNeck(i==0 and 64 or 128,128,i==0 and 2 or 1,32))
            groups2.append(BottleNeck(i==0 and 128 or 256,256,i==0 and 2 or 1,64))
            groups3.append(BottleNeck(i==0 and 256 or 512,512,i==0 and 2 or 1,64))
            groups4.append(BottleNeck(i==0 and 512 or 1024,1024,i==0 and 2 or 1,64))
        self.groups1=nn.Sequential(*groups1)
        self.groups2=nn.Sequential(*groups2)
        self.groups3=nn.Sequential(*groups3)
        self.groups4=nn.Sequential(*groups4)
        self.FC=nn.Linear(1024,10)
    def forward(self,x):
        out=self.conv(x)
        out=self.groups1(out)
        out=self.groups2(out)
        out=self.groups3(out)
        out=self.groups4(out)
        out=F.adaptive_avg_pool2d(out,[1,1])
        out=out.view(out.size()[0],-1)
        out=self.FC(out)
        return out
