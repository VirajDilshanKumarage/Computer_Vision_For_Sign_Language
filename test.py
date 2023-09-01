import math
import time


import cv2

import numpy as np
from mediapipe.python.solutions import hands


cap = cv2.VideoCapture(0)
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
imgSize=300
offset=20
counter=0

lables = ["A","B","C"]
word = []



while True:
    success, img = cap.read()
    imgOutput=img.copy()
    hands,img = detector.findHands(img)
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
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction, index= classifier.getPrediction(imgWhite,draw=False)
            print(prediction,index)

            if index == 0:
                word.append("A")
            if index == 1:
                word.append("B")
            if index == 2:
                word.append("C")





        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize,hCal))
            imgResizeShape = imgResize.shape

            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap,:] = imgResize
            prediction, index = classifier.getPrediction(imgWhite,draw=False)

        cv2.rectangle(imgOutput,(x-offset,y-offset-50),(x-offset+150,y-offset-50+50),(255,0,255),cv2.FILLED)
        cv2.putText(imgOutput,lables[index],(x,y-25),cv2.FONT_HERSHEY_COMPLEX,1.5,(255,255,255),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)
        cv2.imshow("imageCrop",imgCrop)
        cv2.imshow("ImageWhite", imgWhite)


    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)






