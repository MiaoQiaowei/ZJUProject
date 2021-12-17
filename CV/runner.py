from getData import *
from cnn import *
from svm import *
from BoW import *
from torch.utils.data import DataLoader

class Runner:
    def __init__(self):
        self.svm = None
        self.resnet18 = ResNet18(10)
        self.bagOfWords = None
        self.train_set, self.test_set = get_train_test_set()

    
    def train(self, run='cnn'):
        if run == 'cnn':
            go = 1
        
        elif run == 'svm':
            go = 2
        else:
            go =3 
        return 1
    
    def train_cnn(self):
        self.resnet18.cuda()
        train_loader = DataLoader(self.train_set,batch_size=64,shuffle=True)
        test_loader = DataLoader(self.test_set, batch_size=64, shuffle=True)
        opt = torch.optim.Adam(self.resnet18.parameters,lr=0.001)
        loss_func = F.cross_entropy
        for epoch in range(100):
            self.resnet18.train()
            for index, data in enumerate(train_loader):
                logits = self.resnet18(data[0].cuda())
                loss = loss_func(logits,data[1].cuda())
                opt.zero_grad()
                loss.backward()
                opt.step()
                acc = self.get_acc(logits,data[1].cuda())
                print(f'loss:{loss.item()} acc:{acc}')
            
            self.resnet18.eval()
            acc = 0.0
            num = 0
            for index, data in enumerate(train_loader):
                logits = self.resnet18(data[0].cuda())
                loss = loss_func(logits,data[1].cuda())
                opt.zero_grad()
                loss.backward()
                opt.step()
                acc += self.get_acc(logits, data[1].cuda())
                num+=1
            print(f'final acc:{acc/num}')
    
    def get_acc(logits, label):
        label_ = torch.argmax(logits,dim=-1)
        same = (label==label_)
        acc = torch.mean(same)
        return acc