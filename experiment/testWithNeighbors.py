
"""
python testWithNeighbors.py trung_color.jpg
"""

from faceLibs import *
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import cv2


path = "../dataset"

# Change range of Neighbors to test
RANGE_NEIGHBORS = range(13)
# Choose Radius and Cells
RADIUS = 3
CELLVALUE = 7


def getInput():
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    fd = FaceDetector()
    return img, fd


def resultTestWithNeighbors(faceEQ, faces, labels):
    ids = []
    confs = []
    start = time.time()
    for neighbors in RANGE_NEIGHBORS:
        recognizer = cv2.face.LBPHFaceRecognizer_create(RADIUS,
                                                        neighbors,
                                                        CELLVALUE,
                                                        CELLVALUE)
        print(f"TRAINING FOR {neighbors} NEIGHBORS")
        recognizer.train(faces, labels)
        print('LBPH FACE RECOGNISER TRAINED')
        id, conf = recognizer.predict(faceEQ)
        ids.append(id)
        confs.append(round(100-conf, 2))
        print(f"Radius: {RADIUS}, Neighbors: {neighbors}", end=" ")
        print(f"and cells: {CELLVALUE}x{CELLVALUE}")
        print(f"ID is: {id} THE CONFIDENCE: {round(100-conf, 2)}\n")
    end = time.time()
    print(f"Times passed: {round(end - start, 2)}s")
    return ids, confs, neighbors


def drawChart(ids, confs, x_axis):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    plt.subplots_adjust(hspace=0.5)

    ax1.set_ylabel("ID")
    ax1.plot(ids, color='b', marker="o", ms=8, lw=2, mfc='r')
    ax1.grid()
    ax1.set_xlim(xmin=0, xmax=x_axis)

    ax2.set_title(f"With Radius: {RADIUS}, Cells: {CELLVALUE}x{CELLVALUE}")
    ax2.set_ylabel("CONFIDENCE")
    ax2.set_xlabel("NEIGHBORS")
    ax2.plot(confs, color='g', marker="o", ms=8, lw=2, mfc='orange')
    ax2.grid()
    ax2.set_xlim(xmin=0, xmax=x_axis)

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

        ids, confs, neighbors = resultTestWithNeighbors(faceEQ, faces, labels)
        drawChart(ids, confs, neighbors)
