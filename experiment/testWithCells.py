
"""
python testWithCells.py trung_color.jpg
"""

from faceLibs import *
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import cv2


path = "../dataset"

# Change range of Cells to test
RANGE_CELLVALUE = range(1, 11)
# Choose Radius and Neighbors
RADIUS = 3
NEIGHBORS = 9


def getInput():
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    fd = FaceDetector()
    return img, fd


def resultTestWithCells(faceEQ, faces, labels):
    ids = []
    confs = [0]
    start = time.time()
    for cellValue in RANGE_CELLVALUE:
        recognizer = cv2.face.LBPHFaceRecognizer_create(RADIUS,
                                                        NEIGHBORS,
                                                        cellValue,
                                                        cellValue)
        print(f"TRAINING FOR {cellValue}x{cellValue} CELLS")
        recognizer.train(faces, labels)
        print('LBPH FACE RECOGNISER TRAINED')
        id, conf = recognizer.predict(faceEQ)
        ids.append(id)
        confs.append(round(100-conf, 2))
        print(f"Radius: {RADIUS}, Neighbors: {NEIGHBORS}", end=" ")
        print(f"and cells: {cellValue}x{cellValue}")
        print(f"ID is: {id} THE CONFIDENCE: {round(100-conf, 2)}\n")
    end = time.time()
    print(f"Time passed: {round(end - start, 2)}s")
    return ids, confs, cellValue


def drawChart(ids, confs, x_axis):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    plt.subplots_adjust(hspace=0.5)

    ax1.set_ylabel("ID")
    ax1.plot(ids, color='b', marker="o", ms=8, lw=2, mfc='r')
    ax1.grid()
    ax1.set_xlim(xmin=0, xmax=x_axis)

    ax2.set_title(f"With Radius: {RADIUS}, Neighbors: {NEIGHBORS}")
    ax2.set_ylabel("CONFIDENCE")
    ax2.set_xlabel("CELLS")
    ax2.plot(confs, color='g', marker="o", ms=8, lw=2, mfc='orange')
    ax2.grid()
    ax2.set_xlim(xmin=1, xmax=x_axis)

    plt.show()
    return None


if __name__ == "__main__":

    img, fd = getInput()
    faceRect = fd.faceDetect(img)
    faces, labels = getFaceWithID(path)

    for (x, y, w, h) in faceRect:
        faceROI = img[y+(h//9):(y+h)-(h//12), x+(w//7):(x+w)-(w//7)]
        faceROI = resize(faceROI, "w", 150)
        faceEQ = cv2.equalizeHist(faceROI)

        ids, confs, cellValue = resultTestWithCells(faceEQ, faces, labels)
        drawChart(ids, confs, cellValue)
