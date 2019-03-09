import os
import cv2
import glob
import random
import argparse
import cv_tools
import dlib_tools
import numpy as np


class FaceRecognition:
    """
    面部识别, 通过获取摄像头/图片 中的人物和已有的数据对比确定是否是同一个人
    """
    def __init__(self, csv_path):
        """
        param csv_path: 人脸特征csv文件保存路径
        """
        self.csv_path = csv_path
        self.all_load_feature = []
        self.detector = dlib_tools.get_detector()
        self.predictor = dlib_tools.get_predictor()
        self.model = dlib_tools.get_face_model()
        self.load_face_feature()

    def face_compare(self, face_image):
        """
        """
        rgb_image = cv_tools.bgr2rgb(face_image)
        im_shape = rgb_image.shape
        im_height = im_shape[0]
        im_width = im_shape[1]
        faces = self.detector(rgb_image, 1)
        if len(faces) == 0:
            cv_tools.putText(face_image, "No face was detected!", color=(0, 0, 255))
        else:
            feature = self.face_feature(rgb_image, faces)
            csv_feature = random.choice(self.all_load_feature)
            same = self.face_recogition(feature, csv_feature)
            if same:
                cv_tools.putText(face_image, "Yes!", location=(int(im_height/2), int(im_width/2)))
            else:
                cv_tools.putText(face_image, "No!", location=(int(im_height/2), int(im_width/2)), color=(0, 0, 255))

    def load_face_feature(self):
        """
        """
        all_csv = glob.glob(self.csv_path + '/*.csv')
        for filename in all_csv:
            array = np.loadtxt(filename, delimiter=',')
            print("loading data %s" % filename)
            self.all_load_feature.append(array)

    def read_face_from_image(self, image):
        """
        """
        imdata = cv2.imread(image)
        self.face_compare(imdata)
        cv2.imshow("face recogition", imdata)
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()

    def read_face_from_camera0(self):
        """
        """
        video = cv_tools.read_camera0()
        for frame in video:
            self.face_compare(frame)
            cv2.imshow("face recogition", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        cv2.destroyAllWindows()

    def face_feature(self, image, faces):
        """
        """
        array = np.array([])
        shape = self.predictor(image, faces[0])
        face_desc = self.model.compute_face_descriptor(image, shape)
        for _, desc in enumerate(face_desc):
            array = np.append(array, desc)
        return array

    def face_recogition(self, face1, face2):
        """
        """
        # 计算两张脸的欧式距离 方法 1
        # distance = 0
        # for i in range(len(face1)):
        #     distance += (face1[i] - face2[i])**2
        # distance = np.sqrt(distance)
        # 计算两张脸的欧式距离 方法 2
        # distance = np.sqrt(np.sum(np.square(face1 - face2)))
        # 计算两张脸的欧式距离 方法 3
        distance = np.linalg.norm(face1- face2)
        print('euclidean metric:%s' % distance)
        if(distance < 0.5):
            return True
        else:
            return False

    def run(self):
        """
        """
        parse = argparse.ArgumentParser(description="face recognition tools")
        parse.add_argument('-t', '--type', required=True, choices=['0', '1'], help="face recognition type: 0(image), 1(camera)")
        parse.add_argument('-i', '--image', required=False, help="image path")
        args = parse.parse_args()
        print(args.type)
        if args.type == '0':
            img = args.image
            if not img:
                print("face register from image need a image not None")
                return
            if not os.path.exists(img):
                print("image not exists")
                return
            else:
                self.read_face_from_image(img)
        else:
            self.read_face_from_camera0()




if __name__ == '__main__':
    path = 'feature'
    # image = '/home/fantasy/faces/sao.jpg'
    image = '/home/fantasy/faces/james.jpeg'
    reco = FaceRecognition(path)
    # reco.read_face_from_camera0()
    # reco.read_face_from_image(image)
    reco.run()

