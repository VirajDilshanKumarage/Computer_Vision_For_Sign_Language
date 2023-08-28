import math

import cv2

import numpy as np
from mediapipe.python.solutions import hands

cap = cv2.VideoCapture(0)
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
imgSize=400
offset=20
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h=hand['bbox']

        imgWhite = np.ones((imgSize, imgSize,3),np.uint8)*255
        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset]

        imgCropShape=imgCrop.shape

        aspectRatio=h/w
        if aspectRatio>1:
            k=imgSize/h
            wCal= math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape=imgResize.shape
            imgWhite[0:imgCropShape[0], 0:imgCropShape[1]] = imgCrop

        cv2.imshow("imageCrop",imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", img)
    cv2.waitKey(1)



