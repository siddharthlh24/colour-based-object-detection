import numpy as np
import cv2
from matplotlib import pyplot as plt

shapes = ["triangle","quadrilateral","pentagon","hexagon","heptagon","octagon"]

lower = {'red':(0, 5, 5), 'green':(50, 100, 100), 'blue':(80,50,20),}
upper = {'red':(100,255,255), 'green':(86,255,255), 'blue':(150,255,255),}

cap = cv2.VideoCapture(1)
while(True):
    try:
        
        ret, frame = cap.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #kernel = np.ones((5,5),np.float32)/25
        #dst = cv2.filter2D(hsv,-1,kernel)
        #hsv=dst

        blur = cv2.blur(hsv,(20,20))
        #blur = cv2.GaussianBlur(hsv,(5,5),10)
        #median = cv2.medianBlur(hsv,5)
        hsv=blur

        kernel = np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, kernel)
        hsv=closing

        """lower_blue = np.array([100,50,100])
        upper_blue = np.array([150,255,255])

        lower_blue = np.array([100,50,100])
        upper_blue = np.array([150,255,255])"""

        for key, value in upper.items():

            mask = cv2.inRange(hsv, lower[key], upper[key])

            #edges = cv2.Canny(mask,1000,200)
            #plt.imshow(edges,cmap = 'gray')

            res = cv2.bitwise_and(frame,frame, mask= mask)

            """contours,hierarchy = cv2.findContours(mask, 1, 2)

            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt=contours[max_index]


            #M = cv2.moments(cnt)
            #print(M)
            area = cv2.contourArea(cnt)
            print(area)
            #cx = int(M['m10']/M['m00'])
            #cy = int(M['m01']/M['m00'])

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            #print(len(approx))
            
            #print(cx,cy)
            x,y,w,h = cv2.boundingRect(cnt)
            imgk = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)"""
            cv2.imshow('imgk',res)

            #time.sleep(0.5)
        plt.show()
        #cv2.imshow('frame',frame)
        #cv2.imshow('mask',mask)
        #cv2.imshow('res',res)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        pass


cap.release()
cv2.destroyAllWindows()


