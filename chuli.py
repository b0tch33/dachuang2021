# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:00:27 2021

@author: 66441

仅为一张图片的人脸裁剪及提取特征值，之后与读取图片进行衔接
"""

from PIL import Image
import cv2
import numpy as np
from sklearn.decomposition import PCA

# 将裁剪人脸缩放为z指定像素大小
width, height = [400, 400]
dim = (width, height)

# 这里用读取所有图片的代码取代之
path = "E:/study/dachuang/data_test/10/43088_1925-05-12_1956.jpg"
im = Image.open(path)

# print(im.shape)

im = np.array(im)

face_model = cv2.CascadeClassifier(r'D:/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
# 图片进行灰度处理
im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
# 人脸检测
faces = face_model.detectMultiScale(im)
# 裁剪人脸
for (x,y,w,h) in faces:
    face_im = im[y:y+h, x:x+w]


# 进行缩放
face_im = cv2.resize(face_im, dim, interpolation=cv2.INTER_AREA)

# PCA降维（未完成）
pca = PCA(n_components=20)
face_im_new = pca.fit_transform(face_im)
huanyuan = pca.inverse_transform(face_im_new)
huanyuan = huanyuan.astype(np.uint8)
print(huanyuan)
print(pca.explained_variance_ratio_)
# print(new_image)

# 将图片（这里等于矩阵）存放进data列表，准备进行下一步
data = []
data.append(huanyuan)
# print(data[0])

# # 显示图片
cv2.imshow('image', im)
cv2.imshow('face', face_im)
cv2.imshow('face_new', huanyuan)
cv2.waitKey(0)
cv2.destroyAllWindows()