# face-recogition
python + dlib 进行人脸识别

## python version 
   python 3.4.5

## platform
   linux

## Dependency 
   opencv-python，dlib


### step 1 register face from image or camera 第一步先注册人脸(从图片/摄像头)
    python face_register.py -t 1
    
    register face will generate csv file save human face to a 128 dimensional vector

### step 2 recognition face from image or camera 第二步识别人脸(从图片/摄像头)
    python face_recognition.py -t 1
    
    read human face from image or camera and maps the image of a human face to a 128 dimensional vector
    
 ## register the face from image 从图像中注册面部信息
 ![](https://github.com/pythondever/python-dlib-face-recogition/blob/master/cv/data/faces/harden1.jpeg)
 
 ## recognition 识别结果
 
 ![](https://github.com/pythondever/python-dlib-face-recogition/blob/master/cv/data/faces/cmp1.png)
 
 ![](https://github.com/pythondever/python-dlib-face-recogition/blob/master/cv/data/faces/cmp2.png)
 
    
    
more: https://blog.csdn.net/lucky404/article/details/88184350
