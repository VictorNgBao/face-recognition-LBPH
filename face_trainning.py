#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Copy and Paste to Terminal:
            python face_trainning.py
"""

from faceLibs import getFaceWithID
from time import time
import numpy as np
import cv2

PATH_DATA = "dataset"
PATH_TRAIN = "trainning/trainDataLBPH.yml"


def trainningFace():
    recognizer = cv2.face.LBPHFaceRecognizer_create(3, 9, 7, 7)
    print('TRAINING......')
    faces, ids = getFaceWithID(PATH_DATA)
    for id in range(1, len(np.unique(ids))+1):
        start1 = time()
        recognizer.train(faces, ids)
        end1 = time()
        print(f"{id} FACE(S) TRAINED IN '{round(end1-start1, 3)}s' .")
    start2 = time()
    recognizer.write(PATH_TRAIN)
    end2 = time()
    print(f"SAVING IS SUCCESSFUL IN '{round(end2-start2, 3)}s' .")
    print("TRAINNING PROCESS IS COMPLETED!\n")


if __name__ == "__main__":

    trainningFace()
