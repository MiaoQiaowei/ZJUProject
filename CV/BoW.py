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
        train_path_list = os.listdir(train_path)
        # 聚类中心数, 设定为同Mnist样本数目一样
        word_num = 10  

        image_paths = []  # 所有图片路径
        dataset = {}
        for set_name in train_path_list:
            path_list = os.listdir(train_path + "/" + set_name)
            dataset[set_name] = len(path_list)
            for file_path in path_list:
                image_path = os.path.join(train_path + set_name, file_path)
                image_paths += [image_path]

        sift_det=cv2.xfeatures2d.SIFT_create()
        descriptor_list=[]  # 特征描述


        for name, num in dataset.items():
            dir = train_path + name
            print("从 " + name + " 中提取特征")
            for file_name in os.listdir(dir):
                file_name = f'{dir}//{file_name}'
                img=cv2.imread(file_name)
                gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                keypoint,descriptor=sift_det.detectAndCompute(gray,None)
                if descriptor is None:
                    continue
                descriptor_list.append((image_path, descriptor))

        descriptors = descriptor_list[0][1]
        for image_path, descriptor in descriptor_list[1:]:
            descriptors = np.concatenate([descriptors,descriptor],axis=0) 

        print ("开始 k-means 聚类: %d words, %d key points" %(word_num, descriptors.shape[0]))
        voc, variance = kmeans(descriptors, word_num, 1) 

        image_features = np.zeros((len(image_paths), word_num), "float32")
        for i in range(len(image_paths)):
            words, distance = vq(descriptor_list[i][1],voc)
            for w in words:
                image_features[i][w] += 1

        nbr_occurences = np.sum( (image_features > 0) * 1, axis = 0)
        idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

        image_features = image_features*idf
        image_features = preprocessing.normalize(image_features, norm='l2')
        joblib.dump((image_features, image_paths, idf, word_num, voc), "bow.pkl", compress=3)
    
    def predict(self, image_path):
        
        image_features, image_paths, idf, word_num, voc = joblib.load("bow.pkl")
            
        sift_det=cv2.xfeatures2d.SIFT_create()
        descriptor_list = []

        im = cv2.imread(image_path)
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        keypoint, descriptor = sift_det.detectAndCompute(gray, None)

        descriptor_list.append((image_path, descriptor))   
        descriptors = descriptor_list[0][1]

        test_features = np.zeros((1, word_num), "float32")
        if descriptors is None:
            return 0
        words, distance = vq(descriptors,voc)
        for w in words:
            test_features[0][w] += 1

        test_features = test_features*idf
        test_features = preprocessing.normalize(test_features, norm='l2')

        score = np.dot(test_features, image_features.T)
        rank_ID = np.argsort(-score)

        for i in range(1):
            index = rank_ID[0][i]
            predict_image_path = image_paths[index]
            top1_predict_image_label = predict_image_path[-5]
            label = image_path[-5]
            if top1_predict_image_label == label:
                return 1
        return 0

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