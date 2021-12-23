import cv2
import numpy as np
import scipy.spatial.distance as dist
import os
import joblib

#线性核函数
class LinearKernel:
    def __call__(self, x, y):
        return np.dot(x, y.T)

#多项式核函数
class PolyKernel:
    #初始化方法
    def __init__(self, degree=2):
        self.degree = degree
    def __call__(self, x, y):
        return np.dot(x, y.T) ** self.degree

#高斯核函数
class RBF:
    def __init__(self, gamma=0.1):
        self.gamma = gamma
    def __call__(self, x, y):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        return np.exp(-self.gamma * dist.cdist(x, y) ** 2).flatten()

class SVM:
    def __init__(self, kernel, data_num, data_shape,max_iter=500):
        self.C = 0.6
        
        #迭代次数
        self.iter = max_iter
        #权重
        self.W = np.zeros(data_shape)
        #偏置
        self.B = 0
        #拉格朗日算子
        self.alpha =  np.zeros(data_num)
        #核函数
        self.kernel = LinearKernel if kernel is None else kernel
        #保存经过核函数处理过后的结果
        self.kernel_out = np.zeros((data_num,data_num))
        
    def get_index(self, j, data_num):
        index = np.array([i for i in range(data_num) if i != j])
        np.random.shuffle(index)
        return index[0]

    def get_diff(self, index, data, label):
        k_v = self.kernel(data, data[index])
        tmp = self.alpha * label
        pred = np.dot(tmp.T , k_v.T) + self.B
        return pred - label[index]
    
    def get_real_alpha(self, i, j, label):
        if label[i] != label[j]:
            low = max(0, self.alpha[j] - self.alpha[i])
            high = min(self.C, self.C-self.alpha[i]+self.alpha[j])
        else:
            low = max(0, self.alpha[i] + self.alpha[j] - self.C)
            high = min(self.C, self.alpha[i]+self.alpha[j])
        return low, high
    
    def get_new_alpha(self, alpha, low, high):
        if alpha> high:
            return high
        if alpha < low:
            return low
        return alpha

    def fit(self, data, label):
        data_num = data.shape[0]
        data_shape = data.shape[1]
        for i in range(data_shape):
            self.kernel_out[:, i] = self.kernel(data, data[i, :])

        #样本数目
        self.data_num = data_num
        #样本形状
        self.data_shape = data_shape

        for iter in range(self.iter):
            pre_alpha = np.copy(self.alpha)
            for j in range(self.data_num):
                i  = self.get_index(j, self.data_num)
                diff_i = self.get_diff(i,data, label)
                diff_j = self.get_diff(j, data, label)
                if (label[j] * diff_j < -0.001 and self.alpha[j] < self.C) or (label[j] * self.alpha[j] > 0.001 and self.alpha[j] > 0):
                    eta = 2.0*self.kernel_out[i,j] - self.kernel_out[i,i] - self.kernel_out[j,j]

                    if eta >=0 :
                        continue
                    low, high = self.get_real_alpha(i,j, label)
                    old_j = self.alpha[j]
                    old_i = self.alpha[i]
                    self.alpha[j] -= (label[j] * (diff_i-diff_j)) / eta

                    self.alpha[j] = self.get_new_alpha(self.alpha[j], low, high)
                    self.alpha[i] = self.alpha[i] + label[i] * label[j] * (old_j - self.alpha[j])

                    b1 = self.B - diff_i - label[i] * (self.alpha[i] - old_j) * self.kernel_out[i, i] - \
                         label[j] * (self.alpha[j] - old_j) * self.kernel_out[i, j]
                    b2 = self.B - diff_j - label[j] * (self.alpha[j] - old_j) * self.kernel_out[j, j] - \
                         label[i] * (self.alpha[i] - old_i) * self.kernel_out[i, j]
                    if 0 < self.alpha[i] < self.C:
                        self.B = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.B = b2
                    else:
                        self.B = 0.5 * (b1 + b2)
            
            diff = np.linalg.norm(self.alpha - pre_alpha)
            if diff < 1e-3:
                for n in range(data_shape):
                    self.W[n] = np.sum(self.alpha * label * data[:, n])
                break
            
        

    def predict(self, data):
        pred = np.zeros(data.shape[0])
        for i in range(len(data)):
            dd = data[i, :]
            pred[i] = np.sign(np.dot(self.W, dd) + self.B)
        return pred

def load_data(fname):
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)
def eval_acc(label, pred):
    return np.sum(label == pred) / len(pred)

def get_data_label(path):
    data = []
    label = []
    for dir in os.listdir(path):
            dir_path = f'{path}/{dir}'
            for file_path in os.listdir(dir_path):
                file_path = f'{dir_path}/{file_path}'
                label.append(int(file_path[-5]))
                image = cv2.imread(file_path,0)
                image = np.array(image).reshape(-1)
                print(image.shape)
                data.append(image)
    data = np.array(data)
    label = np.array(label)
    return data, label

if __name__ == '__main__':
    x_train,y_train = get_data_label('data\\trainset')
    x_test,y_test = get_data_label('data\\testset')

    # kernel = RBF(gamma=0.1)
    # svm = SVM(kernel=kernel,data_num=22924,data_shape=784)  # 初始化模型
    # from sklearn import svm
    # print('fitting')
    # predictor = svm.SVC(gamma='scale', C=1.0,
    #                     decision_function_shape='ovr', kernel='rbf')

    # predictor.fit(x_train, y_train)
    # joblib.dump(predictor,'svm.pkl')
    svm = joblib.load('./svm.pkl')

    svm.fit(x_train,y_train)  # 训练模型
    y_train_pred = svm.predict(x_train)  # 预测标签
    y_test_pred = svm.predict(x_test)

    # 评估结果，计算准确率
    acc_train = eval_acc(y_train, y_train_pred)
    acc_test = eval_acc(y_test, y_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))