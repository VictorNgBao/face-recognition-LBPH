#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    FaceLibs is a library with some methods are used for
    Face detection and Face recognition
                            (Created by: Nguyen Bao Trung)
"""

from threading import Thread
import numpy as np
import cv2
import os

# These below are Global variables,
# Please don't change (except Path to directory of them change):
FACE_CASCADEPATH = cv2.CascadeClassifier(
                        "cascade/haarcascade_frontalface_default.xml")
EYE_CASCADEPATH = cv2.CascadeClassifier(
                        "cascade/haarcascade_eye.xml")
NAME_LIST = "Names.txt"
# Changing if it is neccessary
FONT = cv2.FONT_HERSHEY_SIMPLEX


class WebcamVideoStream:
    """
    Class is taking from imutils library of Adrian Rosebrock (PyImageSearch)
    """
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.stream.release()


class FaceDetector:
    """
    FaceDetector collects some methods are used for
    Face detection and Face recognition:
        1,faceDetect
        2,eyeDetect
    """
    def __init__(self):
        """ Getting paths of HaarCascade, use for face, eye detection """
        self.faceCascadePath = FACE_CASCADEPATH
        self.eyeCascadePath = EYE_CASCADEPATH

    def faceDetect(self, image, scaleFactor=1.3,
                   minNeighbors=5, minSize=(30, 30)):
        """
        Method is used to detect face
            1,image: An image Matrix with type=unit8
            2,scaleFactor: How much image size is reduced at each image scale
            3,minNeighbors: How many neighbors each candidate
                          rectangle should have. To retain it
            4,minSize: Objects smaller than that are ignored
        """
        self.faceRect = self.faceCascadePath.detectMultiScale(
                                                image,
                                                scaleFactor=scaleFactor,
                                                minNeighbors=minNeighbors,
                                                minSize=minSize,
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        return self.faceRect

    def eyeDetect(self, roiImage, scaleFactor=1.05,
                  minNeighbors=4, minSize=(3, 3)):
        """
        Method is used to detect eyes
            1,image: An image Matrix with type=unit8
            2,scaleFactor: How much image size is reduced at each image scale
            3,minNeighbors: How many neighbors each candidate
                          rectangle should have. To retain it
            4,minSize: Objects smaller than that are ignored
        """
        eyeRect = self.eyeCascadePath.detectMultiScale(
                                        roiImage,
                                        scaleFactor=scaleFactor,
                                        minNeighbors=minNeighbors,
                                        minSize=minSize,
                                        flags=cv2.CASCADE_SCALE_IMAGE)
        return eyeRect


def assignColorPerson(name, specPerson):
    """ Method is used in 'face recog' to assign color for special person """
    if (specPerson is not None) and (name in specPerson):
        color_person = (0, 0, 255)
    else:
        color_person = (10, 255, 10)
    return color_person


def showNameList():
    """ Method is used to return list of names in NAME_LIST """
    n = []
    with open(NAME_LIST, "r") as names:
        for line in names:
            if line != "":
                n.append(line.split(',')[1].strip())
    return n


def addName():
    """ Method is used in 'face capture' to add your name to NAME_LIST """
    name = input('Enter your name: ').lower().strip()
    with open(NAME_LIST, "r+") as info:
        id = sum([1 for line in info])+1
        print(f"{id},{name}", file=info)
        print(f"Your name stored at position '{id}' in Name.txt")
    return id


def findName(id, confidence):
    """
    Method is used to find name with confidence for face
        1,id: ID get from image
        2,confidence: A distance between two histogram,
                        is taken from "predic method" of
                        cv2.face.LBPHFaceRecognizer_create()
    """
    with open(NAME_LIST, "r") as info:
        names = []
        for line in info:
            if line == "":
                break
            names.append(line.split(",")[1].rstrip())
        name = f"{names[id-1]}"
        conf = f"{round(100-confidence)}%"
    return name, conf


def createFolderHoldImage(path_data, folder_image):
    """
    Method is used in 'face capture' to create folder,
        that holds image of each User
    """
    os.mkdir(f'{path_data}/{folder_image}')
    return None


def getFaceWithID(path):
    """
    Method is used in 'face trainning' to get samples of Faces and IDs
        path: link to folder that contains images
              e.g.: "C:/my_project/dataset"
    """
    faces, ids = [], []
    for user_path, many_users, images in os.walk(path):
        for user in many_users:
            image_path = os.path.join(user_path, user)
            for images in os.listdir(image_path):
                gray = cv2.imread(os.path.join(image_path, images),
                                  cv2.IMREAD_GRAYSCALE)
                h, w = gray.shape[:2]
                x, y = 0, 0
                # with image(110x110) => image(84x105)
                img = gray[y:y+h-(h//22), (w//2)-(10*w//26):(w//2)+(10*w//26)]
                img_numpy = np.array(img, dtype='uint8')
                id = int(os.path.split(images)[-1].split(".")[1])
                faces.append(img_numpy)
                ids.append(id)
    return faces, np.array(ids)


def pressQtoQuitCV2():
    """ Method is used to quit program """
    return (cv2.waitKey(1) & 0xFF == ord('q'))


def resize(image, select=None, size=None, inter=cv2.INTER_AREA):
    """
    Method is used to resize image
        1,select: "w"(width) or "h"(heigth)
        2,size: in pixel
        3,interpolation:
            cv2.INTER_AREA (more accurate)
            cv2.INTER_CUBIC
            cv2.INTER_NEAREST
    """
    img_height, img_width = image.shape[:2]
    if select == "w":
        aspect_radio = size / float(img_width)
        dim = (size, int(img_height * aspect_radio))
    if select == "h":
        aspect_radio = size / float(img_height)
        dim = (int(img_width * aspect_radio), size)
    if select is None:
        return image
    # image is resized with these information above
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def faceMask(image):
        """
        Method is used for masking to find ROI
            image: An image Matrix with type=unit8
        """
        # take two coordinates from original image
        h, w = image.shape[:2]
        cX, cY = w // 2, h // 2
        mask = np.zeros((w, h), dtype="uint8")
        cv2.rectangle(mask,
                      (cX-(10*w//26), 0),
                      (cX+(10*w//26), h),
                      255, -1)
        # draw two triangle at the bottom of two sides of image
        pts1 = np.array([[cX-(10*w//26), 10*h//14],
                         [cX-w//5, h],
                         [cX-(10*w//26), h]],
                        np.int32)
        pts1 = pts1.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts1], (0))
        pts2 = np.array([[cX+(10*w//26), 10*h//14],
                         [cX+w//5, h],
                         [cX+(10*w//26), h]],
                        np.int32)
        pts2 = pts2.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts2], (0))
        applyMask = cv2.bitwise_and(image, image, mask=mask)
        return applyMask
