# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:59:25 2021

@author: 66441
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:50:12 2021

@author: 66441
"""

from PIL import Image
import cv2
import os
import warnings
warnings.filterwarnings("error", category=UserWarning)

import _pickle as cPickle
import numpy as np
import threading
import time
import copy
from sklearn.decomposition import PCA

time_start=time.time()

path = "E:/study/dachuang/data_test"
images = []
datas = []

width, height = [400, 400]
dim = (width, height)

# 读取图片线程
def read_img(path, i):
    path1 = path + "/" + str(i)
    for image in os.listdir(path1):
        im = Image.open(path1 + '/' + image)
        # v = 1
        # print(v)
        # if v < 5:
        #     print(im)
        # im = cv2.cvtColor(numpy.asarray(Img_img))
        face_model = cv2.CascadeClassifier(r'D:/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
        # 图片进行灰度处理
        im_gray = im.convert("L")
        im_gray = np.array(im_gray)
        face_im = copy.deepcopy(im_gray)
        # 人脸检测
        faces = face_model.detectMultiScale(im_gray)
        if faces is None:
            continue
        # 裁剪人脸
        for (x,y,w,h) in faces:
            face_im = im_gray[y:y+h, x:x+w]
        
        # 进行缩放
        face_im = cv2.resize(face_im, dim, interpolation=cv2.INTER_AREA)
        
        # PCA降维（未完成）
        pca = PCA(n_components=20)
        face_im_new = pca.fit_transform(face_im)
        huanyuan = pca.inverse_transform(face_im_new)
        huanyuan = huanyuan.astype(np.uint8)
        # print(huanyuan)
        # print(pca.explained_variance_ratio_)
        # print(new_image)
        
        # 将图片（这里等于矩阵）存放进data列表，准备进行下一步
        datas.append(huanyuan)
        # print("图片数量：", v)
        # v += 1
        # im.close()
        im.close()
        
# 创建线程
threads = []
x = 1
for t in range(0, 105):
    t = threading.Thread(target=read_img, args=(path, x))
    threads.append(t)
    x += 1
    
# print(threads)

# 未使用的pkl文件方案，先预留不做删除
# with open('images0.pkl','ab') as fp:
#     for image in images:
#         im = Image.open(image)
#         cPickle.dump(im, fp, -1)
#         im.close()
# # with open('images.pkl','ab') as fp:
# #     im = Image.open(path2 + '/' + image)
# #     # images.append(im)
# #     cPickle.dump(im, fp, -1)
# #     # print(np.array(im))
# #     im.close()
# read_file=open('images0.pkl','rb')  
# images = cPickle.load(read_file)
# images.show()
# print("图片数量：", images)

# 主函数，调用线程
if __name__=="__main__":
    for thr in threads:
        thr.start()
    for i in range(10):
        threads[i].join()
    
    print("图片数量：", len(datas))
    
    # for i in range(100):
    #     print(images[i])
        
    time_end=time.time()
print('totally cost',time_end-time_start)
    
    # v = 1
    
    # for im in images:
    #     print(v)
    #     im_t = np.array(copy.deepcopy(im))
    #     if v < 5:
    #         print(im_t)
    #     # im = cv2.cvtColor(numpy.asarray(Img_img))
    #     face_model = cv2.CascadeClassifier(r'D:/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
    #     # 图片进行灰度处理
    #     im_gray = cv2.cvtColor(im_t, cv2.COLOR_RGB2GRAY)
    #     # 人脸检测
    #     faces = face_model.detectMultiScale(im_gray)
    #     # 裁剪人脸
    #     for (x,y,w,h) in faces:
    #         face_im = im_gray[y:y+h, x:x+w]
        
        
    #     # 进行缩放
    #     face_im = cv2.resize(face_im, dim, interpolation=cv2.INTER_AREA)
        
    #     # PCA降维（未完成）
    #     pca = PCA(n_components=20)
    #     face_im_new = pca.fit_transform(face_im)
    #     huanyuan = pca.inverse_transform(face_im_new)
    #     huanyuan = huanyuan.astype(np.uint8)
    #     # print(huanyuan)
    #     # print(pca.explained_variance_ratio_)
    #     # print(new_image)
        
    #     # 将图片（这里等于矩阵）存放进data列表，准备进行下一步
    #     datas.append(huanyuan)
    #     v += 1
    # print("图片数量：", len(datas))