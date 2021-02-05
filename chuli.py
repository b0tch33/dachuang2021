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
import copy
import uuid
uuid_str = uuid.uuid4().hex

import warnings
warnings.filterwarnings("ignore")

# 将裁剪人脸缩放为z指定像素大小
width, height = [128, 128]
dim = (width, height)

# 这里用读取所有图片的代码取代之
path = "E:/study/dachuang/data_test/10/42621_1972-09-27_2011.jpg"
im = Image.open(path)

# print(im.shape)

# im = np.array(im)
# im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
print(type(im))

face_model = cv2.CascadeClassifier(r'D:/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
# 图片进行灰度处理
im = im.convert('L')
im_gray = np.array(im)
face_im = copy.deepcopy(im_gray)
# 人脸检测
faces = face_model.detectMultiScale(im_gray)
print(faces)
print(faces == ())
# 裁剪人脸
for (x,y,w,h) in faces:
    face_im = im_gray[y:y+h, x:x+w]


# 进行缩放
face_im = cv2.resize(face_im, dim, interpolation=cv2.INTER_AREA)

# PCA降维（未完成）
pca = PCA(n_components=8)
face_im_new = pca.fit_transform(face_im)
huanyuan = pca.inverse_transform(face_im_new)
huanyuan = huanyuan.astype(np.uint8)
print(huanyuan.shape)
print(face_im_new.shape)
print(pca.explained_variance_ratio_)
np.save("test.npy", huanyuan)
np.save("test2.npy", face_im)
# print(new_image)

# 将图片（这里等于矩阵）存放进data列表，准备进行下一步
data = []
data.append(huanyuan)
# print(data[0])

name = uuid_str + ".jpg"
cv2.imwrite(name, huanyuan)
cv2.imwrite("test2.jpg", face_im)


# # 显示图片
cv2.imshow('image', im_gray)
cv2.imshow('face', face_im)
cv2.imshow('face_new', huanyuan)
cv2.waitKey(0)
cv2.destroyAllWindows()