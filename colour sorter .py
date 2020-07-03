import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

shapes = ["triangle","quadrilateral","pentagon","hexagon","heptagon","octagon"]

#lower = { 'blue':(100,100,100),'green':(40, 40, 100),'orange':(5, 40, 100),'red':(166, 84, 141) }
#upper = { 'blue':(140,255,255),'green':(60,255,255),'orange':(20,255,255),'red':(186,255,255)}
lower = {'red':(166, 84, 141), 'green':(40, 40, 100), 'blue':(97, 100, 117), 'yellow':(23, 59, 119), 'orange':(0, 50, 80)} #assign new item lower['blue'] = (93, 10, 0)
upper = {'red':(186,255,255), 'green':(60,255,255), 'blue':(117,255,255), 'yellow':(54,255,255), 'orange':(20,255,255)}

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (50,50)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

img_width, img_height = 400,300
dim = (img_width, img_height)

cap = cv2.VideoCapture(1)
while(True):
    ret, frame = cap.read()
    frame = cv2.convertScaleAbs(frame, alpha=1, beta=-20)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hsv = cv2.resize(hsv, dim, interpolation = cv2.INTER_AREA)

    #kernel = np.ones((5,5),np.float32)/25
    #dst = cv2.filter2D(hsv,-1,kernel)
    #hsv=dst

    blur = cv2.blur(hsv,(3,3))
    #blur = cv2.GaussianBlur(hsv,(2,2),3)
    #median = cv2.medianBlur(hsv,5)
    hsv=blur

    kernel = np.ones((5,5),np.uint8)
    #closing = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, kernel)
    #hsv=closing

    """lower_blue = np.array([100,50,100])
    upper_blue = np.array([150,255,255])

    lower_blue = np.array([100,50,100])
    upper_blue = np.array([150,255,255])"""

    for key, value in upper.items():

        mask = cv2.inRange(hsv, lower[key], upper[key])
        cv2.imshow('mask',mask)

        #edges = cv2.Canny(mask,1000,200)
        #plt.imshow(edges,cmap = 'gray')

        #res = cv2.bitwise_and(frame,frame, mask= mask)

        contours,hierarchy = cv2.findContours(mask, 1, 2)

        areas = [cv2.contourArea(c) for c in contours]
        try:
            max_index = np.argmax(areas)
            cnt=contours[max_index]
            #M = cv2.moments(cnt)
            #print(M)
            
            area = cv2.contourArea(cnt)
            #print(area)
            #cx = int(M['m10']/M['m00'])
            #cy = int(M['m01']/M['m00'])
            if(area>2000):
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

                print(len(approx))
                if (len(approx)<9):
                    #print(key , shapes[len(approx)-3])
                    vgx = str(str(key) + (" ") + str(shapes[len(approx)-3]))
                    print(vgx)
                
                #print(cx,cy)
                
                x,y,w,h = cv2.boundingRect(cnt)
                imgk = cv2.rectangle(hsv,(x,y),(x+w,y+h),(0,255,0),2)
                imgk = cv2.resize(imgk, (640,480), interpolation = cv2.INTER_AREA)
                cv2.putText(imgk,vgx,bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
        except:
            pass

        time.sleep(0.1)
        try:
            cv2.imshow('hsv form',imgk)
            cv2.imshow('regular image',frame)
        except:
            cv2.imshow('hsv form',hsv)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)}
    


cap.release()
cv2.destroyAllWindows()


