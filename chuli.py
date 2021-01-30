# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:00:27 2021

@author: 66441

仅为一张图片的人脸裁剪，之后与读取图片进行衔接
"""

import cv2
import numpy as np
from sklearn.decomposition import PCA

# 将裁剪人脸缩放为z指定像素大小
width, height = [400, 400]
dim = (width, height)

def tezhenglian(num):
    new_image = face_im_255_new[:,num]
    # new_image = new_image.astype(np.uint8)
    new_image = scale(new_image) * 255
    new_image = new_image.astype(np.uint8)
    print(new_image)
    new_image = new_image.reshape(20,20)
    new_image = cv2.resize(new_image, dim, interpolation=cv2.INTER_AREA)
    # print(type(new_image))
    return new_image

# 归一化
def scale(A):
    return (A-np.min(A))/(np.max(A) - np.min(A))

# 这里用读取所有图片的代码取代之
path = "E:/study/dachuang/data_test/10/43088_1925-05-12_1956.jpg"
im = cv2.imread(path)

print(im.shape)

face_model = cv2.CascadeClassifier(r'D:/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
# 图片进行灰度处理
im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
# 人脸检测
faces = face_model.detectMultiScale(im)
# print(faces)
# 裁剪人脸
for (x,y,w,h) in faces:
    #1.原始图片；2坐标点；3.矩形宽高 4.颜色值(RGB)；5.线框
    # cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 2)
    # print(x,y,w,h)
    face_im = im[y:y+h, x:x+w]


# 进行缩放
face_im_255 = cv2.resize(face_im, dim, interpolation=cv2.INTER_AREA)

# PCA降维（未完成）
pca = PCA(n_components=100)
face_im_255_new = pca.fit_transform(face_im_255)
# print(face_im_255_new.shape)
# new_image = np.around(face_im_255_new)
temp_im = np.zeros((400,400), dtype=np.uint8)
temp_im += 255
# print(temp_im.shape)
for i in range(100):
    temp_im = temp_im ^ tezhenglian(i)
    print(i)
    cv2.imshow('face_new', temp_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
print("finish")
# print(new_image)

# 将图片（这里等于矩阵）存放进data列表，准备进行下一步
data = []
# data.append(new_image)
# print(data[0])

# # 显示图片
# cv2.imshow('image', im)
# cv2.imshow('face', face_im_255)
# cv2.imshow('face_0', new_image)
zuhe_im = face_im_255 ^ temp_im
cv2.imshow('face_new', zuhe_im)
cv2.waitKey(0)
cv2.destroyAllWindows()