import cv2
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import shutil


path = 'D:/CODE/ZJUProject/CV/mnist'
def get_train_test_set():
    trainset = torchvision.datasets.MNIST( 
        root=path,
        train=True, 
        transform=transforms.ToTensor(),
        download=False
    )

    testset = torchvision.datasets.MNIST(
        root=path,
        train=False,
        transform=transforms.ToTensor(),
        download=False
    )
    return trainset, testset

def save_image():
    trainset, testset = get_train_test_set()
    train_loader = DataLoader(trainset,batch_size=1)
    test_loader = DataLoader(testset,batch_size=1)

    train_path = f'{path}/train'
    for i ,data in enumerate(train_loader):
        image=data[0].numpy()
        label=data[1].numpy()
        file_name = f'{train_path}/{i}_{label[0]}.jpg'
        print(file_name)
        cv2.imwrite(file_name,image)
    
    test_path = f'{path}/test'
    for i ,data in enumerate(test_loader):
        image=data[0].numpy()
        label=data[1].numpy()
        file_name = f'{test_path}/{i}_{label[0]}.jpg'
        print(file_name)
        cv2.imwrite(file_name,image)

def trans2dir(path):
    for name in os.listdir(path):
        print(name)
        if '.jpg' in name:
            label = name[-5]
            print(label)
            srcfile = f'{path}\\{name}'
            dstfile = f'{path}\\{label}\\{name}'
            shutil.move(srcfile, dstfile) 

trans2dir('D:\\CODE\\ZJUProject\\CV\\trainset')
trans2dir('D:\\CODE\\ZJUProject\\CV\\testset')