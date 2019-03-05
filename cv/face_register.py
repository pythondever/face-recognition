import os
import cv2
import dlib
import shutil
import cv_tools
from pathlib import Path
from face_128D import FaceFeatureTo128D


class FaceRegisterTool:
    """
    注册人脸,保存到地址的路径
    """

    def __init__(self, save_path):
        """
        param: save_path 人脸图片保存路径
        """
        self.save_path = save_path
        self.detector = dlib.get_frontal_face_detector()
        self.face128D = FaceFeatureTo128D(self.save_path)
        self.prepare()


    def prepare(self):
        """
        注册之前先清理 save_path 目录确保下面没有其他文件
        """
        path = Path(self.save_path)
        if not path.exists():
            os.makedirs(self.save_path)
        else:
            print('rm %s' % self.save_path)
            shutil.rmtree(self.save_path)
            os.makedirs(self.save_path)
            

    def has_faces(self, image, data):
        """
        param:image
        param:data
        """
        left = 0
        rigth = 0
        bottom = 0
        top = 0
        faces = len(data)
        if not faces:
            cv_tools.putText(image, "No face was detected!", color=(0, 0, 255))
        for _, face in enumerate(data):
            # 拿到面部的位置
            left = face.left()
            rigth = face.right()
            top = face.top()
            bottom = face.bottom()
            # rectangle函数 绘制面部的区域
            cv2.rectangle(image, pt1=(left, top), pt2=(rigth, bottom), color=(0, 255, 0), thickness=2)

    def add_face_from_image(self, image):
        """
        注册人脸(图片方式)
        """
        imdata = cv2.imread(image)
        rgb_image = cv_tools.bgr2rgb(imdata)
        faces = self.detector(rgb_image, 1)
        if len(faces) == 0:
            print("没有检测到人脸: %s" % image)

        else:
            shutil.copy2(image, self.save_path)
            self.face128D.faces_to_128D()

    def add_face_from_camera(self):
        """
        注册人脸(摄像头方式)
        """
        frames = cv_tools.read_camera0()
        count = 0
        for frame in frames:
            image_rgb = cv_tools.bgr2rgb(frame)
            title = 'Register'
            press = cv2.waitKey(2)
            data = self.detector(image_rgb)
            if len(data) == 0:
                cv_tools.putText(frame, "No face was detected!", color=(0, 0, 255))
            if press == ord('q'):
                break
            if press == ord('a'):
                if len(data) == 0:
                    cv_tools.putText(frame, "No face was detected!", color=(0, 0, 255))
                else:
                    count += 1
                    impath = Path(self.save_path).joinpath('%s.jpg' % count)
                    print("保存照片 %s" % impath)
                    cv2.imwrite(str(impath), frame)
            self.has_faces(frame, data)
            cv_tools.putText(frame, 'a:Add', location=(40, 300))
            cv_tools.putText(frame, 'q:Quit', location=(40, 350))
            cv_tools.putText(frame, 'save count:%s' % count, location=(40, 400), size=1.0)
            cv2.imshow(title, frame)
        cv2.destroyAllWindows()
        self.face128D.faces_to_128D()

if __name__ == '__main__':
    save_path = '/home/fantasy/MachineLearning/cv/data/faces'
    img = '/home/fantasy/faces/harden1.jpeg'
    tools = FaceRegisterTool(save_path)
    # tools.add_face_from_camera()
    tools.add_face_from_image(img)