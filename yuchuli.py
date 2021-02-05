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
warnings.filterwarnings("ignore")

import _pickle as cPickle
import numpy as np
import threading
import time
import copy
import uuid
from sklearn.decomposition import PCA

time_start=time.time()

path = "E:/study/dachuang/data_test"
path_jiangwei = "E:/study/dachuang/data_jiangwei"
images = []
datas = []

width, height = [128, 128]
dim = (width, height)

# 读取图片线程
def read_img(path, i):
    global v
    v = 1
    path1 = path + "/" + str(i)
    for image in os.listdir(path1):
        im = Image.open(path1 + '/' + image)
        face_model = cv2.CascadeClassifier(r'D:/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
        # 图片进行灰度处理
        im_gray = im.convert("L")
        im_gray = np.array(im_gray)
        face_im = copy.deepcopy(im_gray)
        # 人脸检测
        faces = face_model.detectMultiScale(im_gray)
        if faces == ():
            continue
        # 裁剪人脸
        for (x,y,w,h) in faces:
            face_im = im_gray[y:y+h, x:x+w]
        
        # 进行缩放
        face_im = cv2.resize(face_im, dim, interpolation=cv2.INTER_AREA)
        
        # PCA降维
        pca = PCA(n_components=8)
        face_im_new = pca.fit_transform(face_im)
        huanyuan = pca.inverse_transform(face_im_new)
        huanyuan = huanyuan.astype(np.uint8)
        # print(huanyuan)
        # print(pca.explained_variance_ratio_)
        # print(new_image)
        
        # 将每一张降维后的图片（矩阵）存为jpg文件，（这个之后做）5000个一组依次存进以1为起始的文件夹内
        uuid_str = uuid.uuid4().hex
        name = path_jiangwei + '/' + uuid_str + ".jpg"
        cv2.imwrite(name, huanyuan)
        print("图片数量：", v)
        v += 1
        im.close()
        
# 创建线程
threads = []
x = 1
for t in range(0, 5):
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

# 主函数，调用线程预处理
if __name__=="__main__":
    v = 1
    for thr in threads:
        thr.start()
    for i in range(5):
        threads[i].join()
        
    time_end=time.time()
print('totally cost',time_end-time_start)