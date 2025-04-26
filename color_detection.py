import cv2
import numpy as np


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
def empty(a):
    pass

cv2.namedWindow('tracker')
cv2.resizeWindow('tracker',640,240)
cv2.createTrackbar("Hue min","tracker",0,179,empty)
cv2.createTrackbar("Hue max","tracker",179,179,empty)
cv2.createTrackbar("sat min","tracker",0,255,empty)
cv2.createTrackbar("sat max","tracker",255,255,empty)
cv2.createTrackbar("val min","tracker",0,255,empty)
cv2.createTrackbar("val max","tracker",255,255,empty)

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    simg = cv2.resize(img,(400,200))
    imgHSV = cv2.cvtColor(simg,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue min","tracker")
    h_max = cv2.getTrackbarPos("Hue max","tracker")
    s_min = cv2.getTrackbarPos("sat min","tracker")
    s_max = cv2.getTrackbarPos("sat max","tracker")
    v_min = cv2.getTrackbarPos("val min","tracker")
    v_max = cv2.getTrackbarPos("val max","tracker")

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower, upper)
    hstack = cv2.bitwise_and(simg,simg,mask=mask)



    cv2.imshow('img',simg)
    # cv2.imshow('imgHSV',imgHSV)
    cv2.imshow('mask',mask)
    cv2.imshow('hstack',hstack)
    # stackImages(0.6,([img,mask,hstack]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

