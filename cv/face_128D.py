import os
import cv2
import shutil
import cv_tools
import dlib_tools
import numpy as np
from imutils import paths


class FaceFeatureTo128D:
    """
    将人脸的信息提取成一个128维的向量空间
    """
    def __init__(self, faces_path):
        self.faces_path = faces_path

    def prepare(self):
        """
        """
        if not os.path.exists('feature'):
            os.mkdir('feature')
        if os.path.exists('feature'):
            shutil.rmtree('feature')
            os.mkdir('feature')

    def get_faces_image(self):
        """
        """
        return paths.list_images(self.faces_path)

    def faces_to_128D(self):
        """
        """
        self.prepare()
        faces = self.get_faces_image()
        detector = dlib_tools.get_detector()
        predictor = dlib_tools.get_predictor()
        face_model = dlib_tools.get_face_model()
        count = 0
        for image in faces:
            imdata = cv2.imread(image)
            image_rgb = cv_tools.bgr2rgb(imdata)
            has_face = detector(image_rgb, 1)
            if len(has_face) == 0:
                print("未检测到人脸 %s" % image)
                continue
            count += 1
            shape = predictor(image_rgb, has_face[0])
            face_desc = face_model.compute_face_descriptor(image_rgb, shape)
            # print('人脸特征: %s' % face_desc)
            feature_array = np.array([])
            for _, desc in enumerate(face_desc):
                feature_array = np.append(feature_array, desc)
            filename = str(count) + '.csv'
            filename = os.path.join('feature', filename)
            print("保存 %s 特征文件 %s" % (image, filename))
            self.face_feature_to_csv(filename, feature_array)
        cv2.destroyAllWindows()

    def face_feature_to_csv(self, filename, col):
        """
        """
        np.savetxt(filename, col, delimiter=',')



if __name__ == "__main__":
    
    faces = "/home/fantasy/MachineLearning/cv/data/faces"
    feature = FaceFeatureTo128D(faces)
    feature.faces_to_128D()