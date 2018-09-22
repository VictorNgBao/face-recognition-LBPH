#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Copy and Paste to Terminal:
            python face_capture.py
"""

from faceLibs import *
import numpy as np
import time
import cv2


PATH_DATA = r'dataset'
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_TEXT = (0, 0, 255)
COLOR_FACE_DETECT = (255, 100, 0)


def getInput2Capture():
    face = FaceDetector()
    faceId = addName()
    camera = WebcamVideoStream(src=0).start()
    return face, faceId, camera


def takePhoto():
    fd, id, cam = getInput2Capture()
    folderImage = f"user {id}"
    createFolderHoldImage(PATH_DATA, folderImage)
    time.sleep(2)
    # Take 30 pictures to dataset
    count = 0
    while count < 30:
        # Condition to quit
        if pressQtoQuitCV2():
            break
        frame = cam.read()
        frame = resize(frame, "w", 320)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Check the luminosity of the frame
        if np.average(gray) > 110:
            faceRect = fd.faceDetect(gray)
            for (x, y, w, h) in faceRect:
                count += 1
                time.sleep(0.3)
                faceROI = gray[y:y+h, x:x+w]
                test = cv2.putText(frame, "FACE DETECTED",
                                   (x-w//4, y-h//4),
                                   FONT, 0.5,
                                   COLOR_FACE_DETECT, 2)
                face = faceMask(faceROI)
                # equalize hist when the light change
                face_equal = cv2.equalizeHist(face)
                cv2.imwrite(f"{PATH_DATA}/{folderImage}/user.{id}.{count}.jpg",
                            resize(face_equal, "w", 110))
                cv2.rectangle(frame,
                              (x+(w//7), y+(h//9)),
                              ((x+w)-(w//7), (y+h)-(h//12)),
                              (0, 255, 0), 2)
                cv2.imshow("Capture photo", face)
            text = "Number of photos: {}".format(count)
            h, w = frame.shape[:2]
            cx, cy = w//2, h//2
            cv2.putText(frame, text,
                        (cx-w//4, cy-2*h//5),
                        FONT, 0.5,
                        COLOR_TEXT, 2)
        cv2.imshow('System is capturing your face...', frame)
    print('CAPTURE FACE PROCESS IS COMPLETED!\n')
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    takePhoto()
