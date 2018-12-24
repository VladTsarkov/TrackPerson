import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt
import math
focal_Length=1.9
real_height=1730
frame_height=990
sensor_height=3.5
def normalize(img,Y):
    shape = img.shape
    y = shape[0]
    x = shape[1]
    Y2=[]
    for i in Y:
        x1 = i[0]
        y1 = i[1]
        x1 = int((x1)/5000)
        y1 = int((y1)/556)
        x1 = int((x1*x+x)/2)
        y1 = int((y1*y+y)/2)
        Y2+=[[x1,y1]]
    return Y2
def k_sred(y):
    n = 5
    Y=[]
    _break = False
    for i in range(len(y)):
        sum=0
        for k in range(n):
            try:
                sum+=y[k+i]
            except IndexError:
                print(i,k)
                _break = True
                break
        if not (_break):
            #y[i]=y[i]/n
            Y+=[y[i]/n]
        else:
            Y+=[y[i]]
    return y

def distance(object_height):
    return focal_Length*real_height*frame_height/object_height/sensor_height

def visualisation(X,T):
    img = cv.imread('Map2.png')
    x1_1,y1_1=0,0
    kx=0
    Y=[]
    Y1_=[]
    Y2_=[]
    #print(len(X))
    for i in range(len(X)):
        x1 = T[i]
        #x1 = X[i][0]
        y1 = X[i][0]
        x1_1 = x1
        y1_1 = y1
        Y+=[[x1,y1]]
        Y1_+=[x1]
        y1=556-y1
        Y2_+=[y1]
        #Y3=k_sred(Y2_)
        #print(x1,y1)
        #cv.circle(img,(x1,y1), 4, (255,0,0), -1)
    """
    Y4=normalize(img, Y)
    for i in Y4:
        cv.circle(img,(i[0],i[1]), 4, (255,0,0), -1)
    print(Y4)
    plt.imshow(img)
    plt.show()
    """
    ll = plt.plot(Y1_,Y2_)
    plt.show()
"""
    print(T)
    for i in range(kx):
        print(len(Y),i)
        Y[len(Y)-i-1][1]=Y[len(Y)-i-1][1]-(kx*100-i*10)
    for i in range(len(Y)):
        cv.circle(img,(Y[i][0],Y[i][1]), 4, (255,0,0), -1)
    print(kx)
    plt.imshow(img)
    plt.show()
    """
cap = cv.VideoCapture('New_one2.mp4')
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
#fgbg = cv.createBackgroundSubtractorMOG2()
tracker = cv.TrackerCSRT_create()
notTrack = True
x1 = 0
y1 = 0
X = []
Y=[]
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    try:
        cnts = cv.findContours(fgmask.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[-2]
    except AttributeError:
        pass
    center = None
    j=0
    if len(cnts) > 0:
        c = max(cnts, key=cv.contourArea)
        rect = cv.minAreaRect(c)
        x,y,w,h = cv.boundingRect(c)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        print("sum",w*h)
        print("distance",distance(h))
        #print(box)
        #cv.drawContours(frame,[box],0,(0,0,255),2)
        if notTrack:
            if w*h>20000:
                boxtrack = (x,y,w,h)
                ok = tracker.init(frame, boxtrack)
                notTrack = False
        else:
            ok, boxtrack = tracker.update(frame)
            Y+=[distance(h)]
            if ok:
                p1 = (int(boxtrack[0]), int(boxtrack[1]))
                p2 = (int(boxtrack[0] + boxtrack[2]), int(boxtrack[1] + boxtrack[3]))
                if x1==0 and y1==0:
                    x1 = int(boxtrack[0])
                    y1 = int(boxtrack[1])
                    X+=[[x1,y1]]
                else:
                    x1 = int(boxtrack[0])
                    y1 = int(boxtrack[1])
                    X+=[[x1,y1]]
                    #print(X)
                    for i in range(len(X)):
                        #print(X[0])
                        try:
                            frame = cv.line(frame, (X[i][0],X[i][1]),(X[i+1][0],X[i+1][1]), (127,127,30), 2)
                        except IndexError:
                            pass
                cv.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            else:
                cv.putText(frame, "Tracking failure detected", (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                notTrack = True
                tracker = cv.TrackerCSRT_create()
                j+=1
                visualisation(X,Y)
                X=[]
    cv.imshow('frame',fgmask)
    cv.imshow('frame2',frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        time.sleep(5)
cap.release()
visualisation(X,Y)
cv.destroyAllWindows()
