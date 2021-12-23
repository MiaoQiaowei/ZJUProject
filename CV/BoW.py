import argparse
import cv2
import joblib
from matplotlib import image
import numpy as np
import os

from pylab import *
from PIL import Image
from scipy.cluster.vq import *
from sklearn import preprocessing

# mpl.rcParams['font.sans-serif'] = ['SimHei']

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", default='data/testset/1/20036_1.jpg')
parser.add_argument("-t", "--train_path", default='data/trainset/')
args = parser.parse_args()

class BoW():
    def __init__(self, args):
        self.train_path = os.getcwd().replace('\\','/')+'/'+args.train_path
        
    
    def save_bow(self, train_path):
        training_names = os.listdir(train_path)
        numWords = 10  # 聚类中心数

        image_paths = []  # 所有图片路径
        ImageSet = {}
        for name in training_names:
            ls = os.listdir(train_path + "/" + name)
            ImageSet[name] = len(ls)
            for training_name in ls[:int(len(ls) / 3)]:
                image_path = os.path.join(train_path + name, training_name)
                image_paths += [image_path]

        sift_det=cv2.xfeatures2d.SIFT_create()
        des_list=[]  # 特征描述


        for name, num in ImageSet.items():
            dir = train_path + name
            print("从 " + name + " 中提取特征")
            for filename in os.listdir(dir):
                filename = f'{dir}//{filename}'
                img=cv2.imread(filename)
                gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                kp,des=sift_det.detectAndCompute(gray,None)
                if des is None:
                    continue
                des_list.append((image_path, des))

        descriptors = des_list[0][1]
        print(descriptors.shape)
        print(des_list[3][1].shape)
        print('生成向量数组')
        for image_path, descriptor in des_list[1:]:
            # descriptors = np.vstack((descriptors, descriptor)) 
            descriptors = np.concatenate([descriptors,descriptor],axis=0) 

        print ("开始 k-means 聚类: %d words, %d key points" %(numWords, descriptors.shape[0]))
        voc, variance = kmeans(descriptors, numWords, 1) 

        im_features = np.zeros((len(image_paths), numWords), "float32")
        for i in range(len(image_paths)):
            words, distance = vq(des_list[i][1],voc)
            for w in words:
                im_features[i][w] += 1

        nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
        idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

        im_features = im_features*idf
        im_features = preprocessing.normalize(im_features, norm='l2')

        print('保存词袋模型文件')
        joblib.dump((im_features, image_paths, idf, numWords, voc), "bow.pkl", compress=3)
    
    def predict(self, image_path):
        
        im_features, image_paths, idf, numWords, voc = joblib.load("bow.pkl")
            
        sift_det=cv2.xfeatures2d.SIFT_create()
        des_list = []

        im = cv2.imread(image_path)
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        kp, des = sift_det.detectAndCompute(gray, None)

        des_list.append((image_path, des))   
        descriptors = des_list[0][1]

        test_features = np.zeros((1, numWords), "float32")
        if descriptors is None:
            return 0
        words, distance = vq(descriptors,voc)
        for w in words:
            test_features[0][w] += 1

        test_features = test_features*idf
        test_features = preprocessing.normalize(test_features, norm='l2')

        score = np.dot(test_features, im_features.T)
        rank_ID = np.argsort(-score)

        for i in range(1):
            index = rank_ID[0][i]
            predict_image_path = image_paths[index]
            top1_predict_image_label = predict_image_path[-5]
            label = image_path[-5]
            if top1_predict_image_label == label:
                return 1
        return 0
    
    def show(self,im ,image_paths, rank_ID):
        figure('基于OpenCV的图像检索')
        subplot(5,5,1)#
        title('目标图片')
        imshow(im[:,:,::-1])
        axis('off')
        print(rank_ID)
        for i, ID in enumerate(rank_ID[0][0:20]):
            print(ID)
            img = Image.open(image_paths[ID])
            subplot(5,5,i+6)
            imshow(img)
            title('第%d相似'%(i+1))
            axis('off')
        show() 

if __name__ =='__main__':
    bow = BoW(args)
    # bow.save_bow(bow.train_path)
    test_path = 'data/testset'
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