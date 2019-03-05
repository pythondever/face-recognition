# python-dlib-face-recogition
python + dlib 进行人脸识别

## python version == 3.4.5 python 版本

## platform linux 操作系统

## Dependency opencv-python，dlib 依赖包


### step 1 register face from image or camera 第一步先注册人脸(从图片/摄像头)
    python face_register.py
    
    register face will generate csv file save human face to a 128 dimensional vector

### step 2 recogition face from image or camera 第二步识别人脸(从图片/摄像头)
    python face_recognition.py
    
    read human face from image or camera and maps the image of a human face to a 128 dimensional vector
    
    
more: https://blog.csdn.net/lucky404/article/details/88184350
