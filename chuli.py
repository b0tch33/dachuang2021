# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:00:27 2021

@author: 66441

仅为一张图片的人脸裁剪，之后与读取图片进行衔接
"""

import cv2
import numpy as np

path = "E:/study/dachuang/data_test/1/10071_1961-04-03_2010.jpg"
im = cv2.imread(path)
face_model = cv2.CascadeClassifier(r'D:/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
# 图片进行灰度处理
gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
# 人脸检测
faces = face_model.detectMultiScale(gray)
# print(faces)
# 裁剪人脸
for (x,y,w,h) in faces:
    #1.原始图片；2坐标点；3.矩形宽高 4.颜色值(RGB)；5.线框
    # cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 2)
    # print(x,y,w,h)
    face_im = im[y:y+h, x:x+w]

# 将裁剪人脸缩放为255*255像素大小
width, height = [255, 255]
dim = (width, height)
# 进行缩放
face_im_255 = cv2.resize(face_im, dim, interpolation=cv2.INTER_AREA)

# 将图片（这里等于矩阵）存放进data列表，准备进行下一步
data = []
data.append(face_im_255)
print(data)

# 显示图片
cv2.imshow('image', im)
cv2.imshow('face', face_im_255)
cv2.waitKey(0)
cv2.destroyAllWindows()