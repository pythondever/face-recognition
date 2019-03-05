import cv2

video = cv2.VideoCapture(0)

def read_camera0():
    """
    读取摄像头
    """

    while 1:
        ok, frame = video.read()
        if not ok:
            print("读取摄像头#0失败")
            return
        else:
            yield frame

    video.release()


def putText(image, text, location=(100, 150), font=cv2.FONT_HERSHEY_COMPLEX, size=1.1, color=(0, 255, 255), font_weight=2):
    """
    往视频上加文字
    param: image 视频/图片
    param: text  文字内容
    param: location 文字的位置
    param: font 字体
    param: size:  字体大小
    param: color: 字体颜色
    param: font_weight 字体粗细
    """
    cv2.putText(image, text, location, font, size, color, font_weight, lineType=cv2.LINE_AA)


def bgr2rgb(image):
    """
    brg 格式转rgb
    """

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
