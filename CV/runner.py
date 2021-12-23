from enum import Flag
from cnn import *
from svm import *
from BoW import *
from torch.utils.data import DataLoader,Dataset
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument( "--train_path", default='data/trainset/')
parser.add_argument("--test_path", default='data/testset/')
args = parser.parse_args()
        

class Runner:
    def __init__(self,args):
        print(f'CV第二次作业')
        self.args = args
    
    def train(self, run='cnn'):
        if run == 'cnn':
            # self.resnet18 = ResNet18(10)
            self.resnet18 = torchvision.models.resnet18(pretrained=False)
            self.resnet18.fc = nn.Linear(512,10)
            self.test_set = Mnist(self.args.test_path)
            self.train_set = Mnist(self.args.train_path)
            self.train_cnn()

        elif run == 'svm':
            x_test,y_test = get_data_label(self.args.test_path)
            x_train,y_train = get_data_label(self.args.train_path)
            svm = joblib.load('./svm.pkl')

            svm.fit(x_train,y_train)
            y_test_pred = svm.predict(x_test)
            acc_test = eval_acc(y_test, y_test_pred)
            print("test accuracy: {:.1f}%".format(acc_test * 100))

        elif run == 'BoW':
            bow = BoW(self.args)
            test_path = self.args.test_path
            acc = 0
            num = 0
            for dir in os.listdir(test_path):
                dir_path = f'{test_path}/{dir}'
                for file_path in os.listdir(dir_path):
                    file_path = f'{dir_path}/{file_path}'
                    print(file_path)
                    acc+= bow.predict(file_path)
                    num+=1

            print(f'acc:{acc/num}')
        else:
            raise ValueError(f'{run} is not support')

    
    def train_cnn(self):
        self.resnet18.cuda()
        train_loader = DataLoader(self.train_set,batch_size=256,shuffle=True)
        test_loader = DataLoader(self.test_set, batch_size=256, shuffle=True)
        opt = torch.optim.Adam(self.resnet18.parameters(),lr=0.005)
        loss_func = F.cross_entropy
        self.resnet18.train()
        counter = 0 
        for epoch in range(5):
            for index, data in enumerate(train_loader):
                counter+=1
                logits = self.resnet18(data[0].cuda())
                opt.zero_grad()
                loss = loss_func(logits,data[1].cuda())
                loss.backward()
                opt.step()
                acc = self.get_acc(logits,data[1].cuda())
                print(f'iter:{counter}-epoch:{epoch}-loss:{loss.item()} acc:{acc}')
            
        self.resnet18.eval()
        acc = 0.0
        num = 0
        for index, data in enumerate(test_loader):
            logits = self.resnet18(data[0].cuda())
            opt.zero_grad()
            loss = loss_func(logits,data[1].cuda())
            loss.backward()
            opt.step()
            acc += self.get_acc(logits, data[1].cuda())
            num+=1
        print(f'final acc:{acc/num}')
    
    def get_acc(self, logits, label):
        label_ = torch.argmax(logits,dim=-1)
        same = (label==label_).float()
        acc = torch.mean(same)
        return acc
if __name__ == '__main__':
    runner = Runner(args)
    runner.train('cnn')
    runner.train('BoW')
    runner.train('svm')