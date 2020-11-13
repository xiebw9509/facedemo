# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:16:14 2018

@author: Administrator

"""
import cv2
import numpy as np
#加载人脸，眼睛和鼻子级联文件
face_cascade=cv2.CascadeClassifier('xl_haarcascade_frontalface_alt.xml')
eye_cascade=cv2.CascadeClassifier('xl_haarcascade_eye.xml')
nose_cascade=cv2.CascadeClassifier('xl_haarcascade_mcs_nose.xml')
#确定级联文件是否正确地加载
#检查脸部级联文件是否加载
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
#检查眼睛级联文件是否加载
if eye_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
#检查鼻子级联文件是否加载
if nose_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
#初始化视频采集对象并定义比例系数
cap=cv2.VideoCapture(0)
#定义比例系数
scaling_factor=0.5
#读取当前帧画面,调整帧大小，转化为灰度图
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,None,fx=scaling_factor,fy=scaling_factor,interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        #在灰度图ROI信息中检测眼睛
        eye_rects=eye_cascade.detectMultiScale(roi_gray)
        #在灰度图ROI信息中检测鼻子
        nose_rects=nose_cascade.detectMultiScale(roi_gray,1.3,5)
        #在眼睛周围画绿色的圈
        for (x_eye,y_eye,w_eye,h_eye) in eye_rects:
            center=(int(x_eye+0.5*w_eye),int(y_eye+0.5*h_eye))
            radius=int(0.3*(w_eye+h_eye))
            color=(0,255,0)
            thickness=3
            cv2.circle(roi_color,center,radius,color,thickness)
        #在鼻子周围画矩形
        for (x_nose,y_nose,w_nose,h_nose) in nose_rects:
            cv2.rectangle(roi_color,(x_nose,y_nose),(x_nose+w_nose,y_nose+h_nose),(0,255,0),3)
            break
    #展示该图像
    cv2.imshow('yuziqing',frame)
    #在下一次迭代之前等待1ms
    #检查是否按了Esc键
    c=cv2.waitKey(1)
    if c==27:
        break
#释放视频采样对象并关闭窗口
cap.release()
cv2.destoryAllWindows()

