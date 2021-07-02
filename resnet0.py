import torch

from torch import nn, optim
from torch.nn import functional as F

from torchvision import datasets
from torchvision import transforms


class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):  # ch_in, ch_out 是通道的数量
        super(ResBlk,self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3,stride= stride,padding=1) #定义第一个卷积层,h,w 图像尺寸保持不变
        self.bn1 = nn.BatchNorm2d(ch_out)  #定义第一个Batch层
        self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.bypass = nn.Sequential()
        if ch_out != ch_in:
            # [b,ch_in,h,w] => [b,ch_out,h,w]
            self.bypass = nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride)
            )
           # print(self.bypass.weight)

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        #print('shape: ', out.shape)
        bypass_out = self.bypass(x)
        #print('bypass_shape: ',bypass_out.shape)
        out = out + bypass_out
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(64)
        )

        self.blk1 = ResBlk(64,128,stride=2) # [b, 64, 16,16] => [b, 128,8,8]
        self.blk2 = ResBlk(128,256,stride=2) # [b, 128, 8,8] => [b, 256,4,4]
        self.blk3 = ResBlk(256,512,stride=2)  # [b, 256, 4,48] => [b, 512,2,2]
        self.blk4 = ResBlk(512,512,stride=1) # [b, 512, 2,2] => [b, 512,2,2]
        self.outlayer = nn.Linear(512,10)  # ch_out * w * h

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = F.adaptive_avg_pool2d(x,[1,1])  # [b,512,2,2] => [b,512,1,1]
        x = x.view(x.size(0),-1)
        x = self.outlayer(x)

        return x

def main():
    x = torch.randn(2,64,32,32)
    ResBasic = ResBlk(64,128,stride=2)
    out = ResBasic(x)
    print('ResBlk Size: ', out.shape)



    y = torch.randn(2,3,32,32)
    model = ResNet18()
    out = model(y)
    print('ResNet18 output size: ', out.shape)
    #print('ResNet18 Paremeters: ', model)

if __name__ == '__main__':
    main()