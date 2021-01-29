# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:35:00 2021

@author: 66441
"""

from PIL import Image
import os
import warnings
warnings.filterwarnings("error", category=UserWarning)

import _pickle as cPickle
import numpy as np
import threading
import time

time_start=time.time()

path = "E:/study/dachuang/data_test"
images = []

# 读取图片线程
def read_img(path, i):
    path1 = path + "/" + str(i)
    for image in os.listdir(path1):
        try:
            im = Image.open(path1 + '/' + image)
            images.append(im)
            im.close()
        except:
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
    
    print("图片数量：", len(images))
    
    for i in range(100):
        print(images[i])
        
time_end=time.time()
print('totally cost',time_end-time_start)

    