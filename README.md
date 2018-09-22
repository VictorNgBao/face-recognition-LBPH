# face-recognition-LBPH
Project: Face Recognition with Rasberry Pi3

## Abstract
The rapid advancement of technology, especially, in the field of computer vision is changing the whole world. Face recognition, the branch of computer vision, is attracting the attention of many people by its applications in surveillance, authentication for banking and security system access, or a lot of personal purposes. The process goes through from collection, detection, pre-processing, and the last stage is recognition.

*Desire to develop a face recognition system that is not expensive, good enough, and ubiquitous in order to serve the demand in the field of surveillance is the mainly purpose of  this work.*

## Introduction
Below are articles and tutorials on the Internet for you to learn about **face recognition**, setup and install **OpenCV with Python**:
* [Face Recognition: Understanding LBPH Algorithm](https://towardsdatascience.com/face-recognition-how-lbph-works-90ec258c3d6b), You should read it first to understand the theory.
* Setup and Install:
  * [Raspbian Stretch: Install OpenCV 3 + Python on your Raspberry Pi](https://www.pyimagesearch.com/2017/09/04/raspbian-stretch-install-opencv-3-python-on-your-raspberry-pi/), You must read it carefully.
  * [Install OpenCV 3 on Windows](https://www.learnopencv.com/install-opencv3-on-windows/), if you need to try and run project on your window machine.
  * **Bonus**: In processing of setup and install, I met a lot of errors and needed much time to solve it. So I give you some resources for your problems:
    * [Warning **: Error retrieving accessibility bus address](https://www.raspberrypi.org/forums/viewtopic.php?t=196070), problem when you run python program on your rasberry pi.
    * [Raspberry Pi IP camera with Motion](http://blog.hawthorn.io/2017/03/07/raspberry-pi-ip-camera-with-motion/), problem with rasberry camera.

## My repository
1. [cascade](https://github.com/VictorNgBao/face-recognition-LBPH/tree/master/cascade) : Folder contains data for face detection.
2. [dataset](https://github.com/VictorNgBao/face-recognition-LBPH/tree/master/dataset) : Folder contains images of users, that was created by program.
3. [experiment](https://github.com/VictorNgBao/face-recognition-LBPH/tree/master/experiment) : Folder contains code to test and draw chart to compare.
4. [trainning](https://github.com/VictorNgBao/face-recognition-LBPH/tree/master/trainning) : Folder contains data of user's face, which were trained.
5. [Names.txt](https://github.com/VictorNgBao/face-recognition-LBPH/blob/master/Names.txt) : File contains names of users, that is got from the program.
6. [faceLibs.py](https://github.com/VictorNgBao/face-recognition-LBPH/blob/master/faceLibs.py) : File contains some methods to call in the whole program.
7. [face_capture.py](https://github.com/VictorNgBao/face-recognition-LBPH/blob/master/face_capture.py) : File contains code to call method that takes some of your pictures, and saves them into **dataset** folder.
8. [face_trainning.py](https://github.com/VictorNgBao/face-recognition-LBPH/blob/master/face_trainning.py) : File contains code to call method that trains your dataset and saves them in a file into **trainning** folder.
9. [face_recog.py](https://github.com/VictorNgBao/face-recognition-LBPH/blob/master/face_recog.py) : File contains code to call method which recognizes faces of users.
10. [start_program.py](https://github.com/VictorNgBao/face-recognition-LBPH/blob/master/start_program.py) : **run it to start program!**
