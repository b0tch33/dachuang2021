# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 14:19:19 2021

@author: 66441
"""

import cv2
import os
import random
import numpy as np
from PIL import Image
from sklearn import neighbors 
from sklearn.cluster import KMeans
import shutil

path = "E:/study/dachuang/data_jiangwei_sex"

train = []
test = []
labels = []
images = os.listdir(path)

for image in os.listdir(path):
    label = image[0]
    # labels.append(label)
    im = Image.open(path + '/' + image)
    im = np.array(im)
    im = im.flatten()
    train.append([im, label])

for i in range(600):
    randIndex = int(np.random.uniform(0, len(train)))
    test.append(train[randIndex])
    del(train[randIndex])
# print(test)

# # print(labels)
knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=2)
knn.fit([i[0] for i in train], [i[1] for i in train]) 
# print(kmeans.labels_)
# print(images[0])
# for idx in range(len(kmeans.labels_)):
#     oldname = path + '/' + images[idx]
#     newname = 'E:/study/dachuang/' + str(kmeans.labels_[idx]) + '/' + images[idx]
#     shutil.copyfile(oldname, newname)
    
res = knn.predict([i[0] for i in test]) 
# print(res)
# print([i[1] for i in test])
error_num = np.sum(res != [i[1] for i in test]) #统计分类错误的数目
num = len(test) #测试集的数目
print("Total num:",num," Wrong num:", error_num,"  准确率:",(num - error_num) / float(num))