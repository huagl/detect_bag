#coding=utf-8
import cv2
import numpy as np
import imutils
import os
import time
from playsound import playsound
import socket
import time
import threading
from threading import Lock

lock = Lock()

bag_cascade = cv2.CascadeClassifier('lbp_xml/cascade.xml')

# 图像缩放
def imgScale(img, width):
    scale = 1.0
    if img.shape[1] > width:
        scale = img.shape[1] / float(width)
        img = imutils.resize(img, width=width)
    return img, scale

def detectBag(img,alpha,beta):
    """
    detectBag(gray, 1.2, 5)
    调节参数，改变检测的精度和速度
    """
    # resize it to reduce detection time
    img, scale = imgScale(img, width=400)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    bags = bag_cascade.detectMultiScale(gray, alpha, beta)
    bags = [(x, y, x+w, y+h) for(x, y, w, h) in bags]
    # mapping to the original image
    ret = [(int(xA * scale), int(yA * scale), int(xB * scale), int(yB * scale)) for (xA, yA, xB, yB) in bags]
    return ret


def drawRectangle(img, boxes, color, thickness):
    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(img, (xA, yA), (xB, yB), color, thickness)
    return img

def disable_zhaji(lock,t,frame):
    lock.acquire()
    playsound('6601.wav')
    global bag_detected_num
    bag_detected_num = bag_detected_num + 1
    cv2.imwrite('/home/hri/gd/taojian/%d.jpg'%bag_detected_num,frame)
    with open('/home/hri/gd/txt/beibao.txt','w') as f:
        f.write(str(bag_detected_num))
    f.close()

    
    c_time=time.time()
    l_time=time.time()
    while(l_time-c_time<t):
        #print l_time-c_time
        l_time=time.time()
        mSocket.sendto("open_door_disable",("192.168.1.213",9293))
        time.sleep(0.2)
        print "send disable"
    lock.release()

def enable_zhaji(lock,t):
    time.sleep(0.01)
    lock.acquire()
    c_time=time.time()
    l_time=time.time()
    while(l_time-c_time<t):
        #print l_time-c_time
        l_time=time.time()
        mSocket.sendto("open_door_enable",("192.168.1.213",9293))
        time.sleep(0.2)
        print "send enable inner"
    lock.release()

if __name__ == "__main__":
    detected_time = -1
    global bag_detected_num
    bag_detected_num=0
    vcap = cv2.VideoCapture("rtsp://192.168.1.115:554/11")
    n_frame = 0

    mSocket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

    while vcap.isOpened():
        ret, frame = vcap.read()
        n_frame += 1
        #  detect every 3 frame
        if ret and n_frame % 3 == 0:
            boxes = detectBag(frame,1.1,4)
            frame = drawRectangle(frame,boxes,(0, 0, 255), 3)
            cv2.imshow("window",frame)

            # 如果当前时刻距离上次检测到背包时刻小于10秒，
            # 则boxes设置为空，不再继续开t1  t2线程
            if time.time() - detected_time>10:
                pass
            else:
                boxes = []

            # 是否有包
            if len(boxes)!=0:
                #检测到背包，记录当前时刻，然后先disable 5秒，再enable 5秒。
                detected_time = time.time()
                t1 = threading.Thread(target=disable_zhaji,args=(lock,5,frame,))
                t1.start()

                t2 = threading.Thread(target=enable_zhaji,args=(lock,5,))
                t2.start()
            else:
                # 真的没包（不在10秒间隙里），则发enable
                if time.time() - detected_time>10:
                    mSocket.sendto("open_door_enable",("192.168.1.213",9293))
                    print "send enable outer"

        # On exit
        if cv2.waitKey(1)&0xff == ord('q'):
            exit(0)
