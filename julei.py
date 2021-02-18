# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 14:19:19 2021

@author: 66441
"""

import cv2
import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import shutil

path = "E:/study/dachuang/data_jiangwei111"

train = []
test = []
images = os.listdir(path)

for image in os.listdir(path):
    im = Image.open(path + '/' + image)
    im = np.array(im)
    im = im.flatten()
    if len(train) <= 1600:
        train.append(im)
    else:
        test.append(im)

# print(train)
kmeans = KMeans(n_clusters=2)
kmeans.fit(train) 
print(kmeans.labels_)
# print(images[0])
for idx in range(len(kmeans.labels_)):
    oldname = path + '/' + images[idx]
    newname = 'E:/study/dachuang/' + str(kmeans.labels_[idx]) + '/' + images[idx]
    shutil.copyfile(oldname, newname)
    
# res = kmeans.predict(test) 
# print(res)