import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt
import math
#cap = cv.VideoCapture('1.mp4')
def visualisation(X):
    img = cv.imread('test.jpg')
    x1_1,y1_1=0,0
    kx=0
    Y=[]
    print(len(X))
    for i in range(len(X)):
        x1 = X[i][0]
        y1 = X[i][1]
        x1 = int(x1/990*1000)
        y1 = 556 -y1
        y1 = int(y1/556*100)
        y1 = 556-y1
        if y1_1!=0 and x1_1!=0:
            #print(x1,x1_1)
            #print(y1,y1_1)
            #print(math.fabs(y1_1-y1))
            if x1<x1_1 and math.fabs(y1_1-y1)<10:
                kx+=1
                #print(kx)
            else:
                if kx>5:
                    print("WHAT")
                kx=0
        x1_1 = x1
        y1_1 = y1
        Y+=[[x1,y1]]
        #print(x1,y1)
        #cv2.circle(img,(x1,y1), 4, (255,0,0), -1)

    print(Y)
    for i in range(kx):
        print(len(Y),i)
        Y[len(Y)-i-1][1]=Y[len(Y)-i-1][1]-(kx-i)
    for i in range(len(Y)):
        cv.circle(img,(Y[i][0],Y[i][1]), 4, (255,0,0), -1)
    print(kx)
    plt.imshow(img)
    plt.show()
#cap = cv.VideoCapture('untitled.mp4')
#cap = cv.VideoCapture('Notus.mp4')
#cap = cv.VideoCapture('2_990.mp4')
#cap = cv.VideoCapture('1_990.mp4')
cap = cv.VideoCapture('NightCam.mp4')
#cap = cv.VideoCapture('StreetCam.mp4')
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
#fgbg = cv.createBackgroundSubtractorMOG2()
tracker = cv.TrackerCSRT_create()
#kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
#fgbg = cv.bgsegm.createBackgroundSubtractorGMG()
notTrack = True
x1 = 0
y1 = 0
X = []
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    #fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

    try:
        cnts = cv.findContours(fgmask.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[-2]
    except AttributeError:
        #cnts=0
        pass
    center = None
    j=0
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv.contourArea)
        #print(c)
        rect = cv.minAreaRect(c)
        x,y,w,h = cv.boundingRect(c)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        print("sum",w*y)
        #print(box)
        cv.drawContours(frame,[box],0,(0,0,255),2)
        if notTrack:
            if w*h>3000:
                boxtrack = (x,y,w,h)
                ok = tracker.init(frame, boxtrack)
                notTrack = False
        else:
            ok, boxtrack = tracker.update(frame)
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
                    print(X)
                    for i in range(len(X)):
                        #print(X[0])
                        try:
                            frame = cv.line(frame, (X[i][0],X[i][1]),(X[i+1][0],X[i+1][1]), (127,127,30), 2)
                        except IndexError:
                            pass
                    #frame = cv.line(frame, (x1,y1),(int(boxtrack[0]),int(boxtrack[1])), (127,127,30), 2)
                cv.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            else:
                cv.putText(frame, "Tracking failure detected", (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                notTrack = True
                tracker = cv.TrackerCSRT_create()
                j+=1
                visualisation(X)
                X=[]
        """
        ((x, y), radius) = cv.minEnclosingCircle(c)
        M = cv.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv.circle(frame, center, 3, (0, 0, 255), -1)
            cv.putText(frame,"centroid", (center[0]+10,center[1]), cv.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255),1)
            cv.putText(frame,"("+str(center[0])+","+str(center[1])+")", (center[0]+10,center[1]+15), cv.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255),1)
        """
    # show the frame to our screen
    #frame = cv.add(frame,mask)

    #cv.namedWindow('frame',cv.WINDOW_NORMAL)
    #cv.resizeWindow('frame',990,540)

    cv.imshow('frame',fgmask)
    cv.imshow('frame2',frame)
    #cv.imshow('frame',)
    k = cv.waitKey(0) & 0xff
    if k == 27:
        time.sleep(5)
cap.release()
visualisation(X)
cv.destroyAllWindows()
