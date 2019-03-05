import dlib

predictor_path = '/home/fantasy/MachineLearning/cv/shape_predictor_5_face_landmarks.dat'
face_model_path = '/home/fantasy/MachineLearning/cv/dlib_face_recognition_resnet_model_v1.dat'


def get_detector():
    """
    """
    return dlib.get_frontal_face_detector()


def get_predictor():
    """
    """
    return dlib.shape_predictor(predictor_path)

def get_face_model():
    """
    """
    return dlib.face_recognition_model_v1(face_model_path)