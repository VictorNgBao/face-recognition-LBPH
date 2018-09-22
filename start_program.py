#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Copy and Paste to Terminal:
            python start_program.py
"""

from face_capture import takePhoto
from face_trainning import trainningFace
from face_recog import findFaceWithName
from faceLibs import showNameList


def printIntro():
    print("{:-<64}".format("-"))
    print("The program for face recognition (created by: Nguyen Bao Trung)")
    print("{:-<64}".format("-"))
    return None


def takeFacePhoto():
    capture = input("\nCapturing your photo to train (Y or N)? ")
    if capture[0].lower() == "y":
        while True:
            takePhoto()
            take_again = input(
                               "Do you want to take more (Y or N)? "
                               ).lower().strip()
            if take_again[0] != "y":
                break
    return None


def trainningFaceFromPhoto():
    trainning = input("\nTrainning machine to recognize face (Y or N)? ")
    if trainning[0].lower() == "y":
        trainningFace()
    return None


def findSpecPerson():
    person = input("\nDo you want to find special person (Y or N)? ")
    if person[0].lower() == "y":
        print("{0:<4}I have some names: {1}".format("", showNameList()))
        print("{:<4}Name of this person: ".format(""), end="")
        who = input("").lower().strip().split(",")
    else:
        who = None
    return who


def recognizeFace():
    who = findSpecPerson()
    recogFace = input("\nRecognizing your face (Y or N)? ")
    if recogFace[0].lower() == "y":
        print("Going to recognized process...")
        findFaceWithName(who)
    return None

if __name__ == "__main__":

    printIntro()
    takeFacePhoto()
    trainningFaceFromPhoto()
    recognizeFace()
