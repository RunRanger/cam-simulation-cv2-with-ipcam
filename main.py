import math
import requests
import cv2
import numpy as np
import imutils

url = "http://192.168.0.184:8081/shot.jpg"

Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 255, 255), (255, 0, 255), (255, 0, 255),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]
SIZE_BALL = 70
HEIGHT_CAMERA = 5.8
WIDTH_CAMERA = 13.3


class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
# print(class_name)
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

#    img_resp = requests.get(url)
#    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#    img = cv2.imdecode(img_arr, -1)


img = cv2.imread('pics/ball_60_10N.jpg')
#img = cv2.imread('pics/ball65-20.jpg')
img = imutils.resize(img, width=1080, height=1080)

widthBall = 0
x = 0
rel = 0
alpha = math.degrees(math.atan(WIDTH_CAMERA/HEIGHT_CAMERA))
heightImg = img.shape[0]
# img = imutils.resize(img, width=416, height=416)
# START ANALYZING
classes, scores, boxes = model.detect(img, Conf_threshold, NMS_threshold)
for (classid, score, box) in zip(classes, scores, boxes):
    color = COLORS[int(classid) % len(COLORS)]
    label = "%s : %f" % (class_name[classid], score)
    cv2.rectangle(img, box, color, 1)
    cv2.putText(img, label, (box[0], box[1]-10),
                cv2.FONT_HERSHEY_COMPLEX, 0.3, color, 1)

    rel = (SIZE_BALL/box[3])
    posXPix = (img.shape[1]/2) - (img.shape[1] - (box[0] + (box[3]/2)))
    x = posXPix * rel

    print("----------")
    yPos = box[1] + box[3]
    relY = ((heightImg - yPos)/(heightImg / 2))

    angle = (90 - alpha) * relY + alpha
    y = math.tan(math.radians(angle)) * HEIGHT_CAMERA
    y2 = math.tan(math.radians(85.90476)) * HEIGHT_CAMERA
    print("----------")
    SENSOR_HEIGHT = 6.8
    FOCAL_LENGTH = 104  # 24
    BALL_HEIGHT = 70

    distance = 0
    heightBallSensor = SENSOR_HEIGHT * (box[3]/heightImg)
    relationH = heightBallSensor/FOCAL_LENGTH
    print(relationH)
    distance = relationH * BALL_HEIGHT
    print(distance)

    break

cv2.imshow("Android_cam", img)
while True:
    # Press Esc key to exit
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
