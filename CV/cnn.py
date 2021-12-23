
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image
import os

class head_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3 ,stride=2,padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel,kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.short_cut_conv = nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=2)
    
    def forward(self, x):
        shortcut = x 
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        shortcut = self.short_cut_conv(shortcut)
        x +=shortcut
        x = self.relu2(x)
        return x
        
class common_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel,kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)

        self.short_cut_conv = nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=1)
    def forward(self, x):
        shortcut = x 
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        shortcut = self.short_cut_conv(shortcut)
        x +=shortcut
        x = self.relu2(x)
        return x

class ResNet18(nn.Module):
    def __init__(self,class_num):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,stride=2,padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        self.layers = nn.Sequential(
            common_block(64,64),
            common_block(64,64),
            common_block(64,64),
            common_block(64,64),

            head_block(64,128),
            common_block(128,128),
            common_block(128,128),
            common_block(128,128),

            head_block(128,256),
            common_block(256,256),
            common_block(256,256),
            common_block(256,256),

            head_block(256,512),
            common_block(512,512),
            common_block(512,512),
            common_block(512,512)
        )
        self.avg_pool = nn.AvgPool2d(7,1)
        self.fc = nn.Linear(512,class_num)
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layers(x)
        x = self.avg_pool(x)
        x = x.view(-1,512)
        x = self.fc(x)
        return x

class Mnist(Dataset):
    def __init__(self,path) -> None:
        super().__init__()
        self.data = []
        for dir in os.listdir(path):
            dir_path = f'{path}/{dir}'
            for file_path in os.listdir(dir_path):
                file_path = f'{dir_path}/{file_path}'
                self.data.append(file_path)
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224)
        ])
    
    def __getitem__(self, index):
        image = Image.open(self.data[index]).convert('RGB')
        image = self.trans(image)
        label = self.data[index][-5]
        return image, int(label)
    def __len__(self):
        return len(self.data)
                



if __name__ == '__main__':
    model = ResNet18(10)
    a = torch.zeros((5,3,224,224))
    out = model(a)
    print(out.shape)