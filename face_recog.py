#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Copy and Paste to Terminal:
            python face_recog.py
"""

from faceLibs import *
import numpy as np
import cv2


PATH_TRAIN = "trainning/trainDataLBPH.yml"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def getInput2Recog():
    fd = FaceDetector()
    camera = WebcamVideoStream(src=0).start()
    recogniser = cv2.face.LBPHFaceRecognizer_create(3, 9, 7, 7)
    return fd, camera, recogniser


def findFaceWithName(specPerson=None):
    fd, cam, recogniser = getInput2Recog()
    recogniser.read(PATH_TRAIN)
    while True:
        # Condition to quit
        if pressQtoQuitCV2():
            break
        frame = cam.read()
        frame = resize(frame, "w", 400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRect = fd.faceDetect(gray)
        for (x, y, w, h) in faceRect:
            faceROI = gray[y+(h//9):(y+h)-(h//12), x+(w//7):(x+w)-(w//7)]
            faceROI = resize(faceROI, "w", 150)
            faceROI_equalHist = cv2.equalizeHist(faceROI)
            id, confidence = recogniser.predict(faceROI_equalHist)
            if confidence < 100:
                name, conf = findName(id, confidence)
                color_name = assignColorPerson(name, specPerson)
                cv2.putText(frame, name,
                            (x-w//3, y),
                            FONT, 0.7, color_name, 2)
                cv2.putText(frame, conf,
                            (x-w//3, y+h//4),
                            FONT, 0.7, color_name, 2)
                cv2.rectangle(frame,
                              (x+(w//7), y+(h//9)),
                              ((x+w)-(w//7), (y+h)-(h//12)),
                              color_name, 2)
            else:
                name = "Unknown"
                conf = "..%"
                color_name = (0, 255, 255)
                cv2.putText(frame, name,
                            (x-w//3, y),
                            FONT, 0.7, color_name, 2)
                cv2.putText(frame, conf,
                            (x-w//3, y+h//4),
                            FONT, 0.7, color_name, 2)
                cv2.rectangle(frame,
                              (x+(w//7), y+(h//9)),
                              ((x+w)-(w//7), (y+h)-(h//12)),
                              color_name, 2)
            cv2.imshow("FaceROI", faceROI_equalHist)
        cv2.imshow("Face", frame)
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    findFaceWithName()
