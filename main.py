import cv2
import numpy as np

from paho.mqtt import client as mqtt_client

# mqtt
broker = '141.94.192.178'
port = 1883
xTopic = "24hours/mydevice/servo/xaxis/command"
yTopic = "24hours/mydevice/servo/yaxis/command"

# opencv
WIDTH = 1280
HEIGHT = 720
SCALE = 0.00392
MINXDIFF = 80  # minimum x difference to move servos
MINYDIFF = 60  # minimum y difference to move servos
V_FOCAL_DEGREES_HALF = 90  # degrees of the camera's field of view
HFOCALDEGREES = 90  # degrees of the camera's field of view
prevXCenter = 0  # previous x coordinate
prevYCenter = 0  # previous y coordinate


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}")
        exit(1)


def connect_mqtt():
    client = mqtt_client.Client()
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def moveX(client, deg):
    result = client.publish(xTopic, deg, qos=0, retain=False)

    # Check if the publish was successful
    status = result.rc
    if status == 0:
        print(f"[MQTT] Send `{result}` to topic `{xTopic}` successfully")
    else:
        print(f"[MQTT] Failed to send message to topic {xTopic}")


def moveY(client, deg):
    result = client.publish(yTopic, deg, qos=0, retain=False)

    # Check if the publish was successful
    status = result.rc
    if status == 0:
        print(f"[MQTT] Send `{result}` to topic `{yTopic}` successfully")
    else:
        print(f"[MQTT] Failed to send message to topic {yTopic}")


print("[MQTT] Connecting to MQTT Broker...")
client = connect_mqtt()

# start client loop to get messages, non-blocking
client.loop_start()

# window ui colors
COLOR = [0, 86, 255]

# read pre-trained model and config file
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")

video = cv2.VideoCapture(1)
video.set(3, WIDTH)  # width
video.set(4, HEIGHT)  # height
video.set(10, 150)  # brightness


# get output layer names
def get_output_layers(net):

    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1]
                         for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1]
                         for i in net.getUnconnectedOutLayers()]

    return output_layers


# draw bounding box
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, center_x, center_y):
    global prevXCenter
    global prevYCenter

    if (class_id == 0):  # person
        label = str("Enemy " + str(round(confidence, 2)))

        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), COLOR, 4)

        cv2.putText(img, label, (x-10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR, 2)

        # move servos
        xDiff = abs(center_x - prevXCenter)
        yDIff = abs(center_y - prevYCenter)

        if (xDiff > MINXDIFF or yDIff > MINYDIFF):
            prevXCenter = center_x
            prevYCenter = center_y

            if (center_x < WIDTH / 2):  # top half of the screen
                center_x = abs((center_x - WIDTH / 2) *
                               HFOCALDEGREES / (WIDTH / 2))
            else:
                center_x = -(center_x - WIDTH / 2) * \
                    HFOCALDEGREES / (WIDTH / 2)

            if (center_y < WIDTH / 2):  # top half of the screen
                center_y = abs((center_y - WIDTH / 2) *
                               V_FOCAL_DEGREES_HALF / (WIDTH / 2))
            else:
                center_y = -(center_y - V_FOCAL_DEGREES_HALF / 2) * \
                    V_FOCAL_DEGREES_HALF / (WIDTH / 2)

            moveX(client, center_x)
            moveY(client, center_y)


# implement YOLO object detection
def get_yolo_objects(img):
    blob = cv2.dnn.blobFromImage(
        img, SCALE, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    centers = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * WIDTH)
                center_y = int(detection[1] * HEIGHT)
                w = int(detection[2] * WIDTH)
                h = int(detection[3] * HEIGHT)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                centers.append([center_x, center_y])

    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        try:
            box = boxes[i]

        except:
            i = i[0]
            box = boxes[i]

        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        center_x = centers[i][0]
        center_y = centers[i][1]

        draw_prediction(img, class_ids[i], confidences[i], round(
            x), round(y), round(x+w), round(y+h), center_x, center_y)

    return img


while True:
    result, video_frame = video.read()  # read frames from the video

    if result is False:
        break  # terminate the loop if the frame is not read successfully

    video_frame = cv2.flip(video_frame, 1)  # flip the video
    video_frame = get_yolo_objects(video_frame)

    cv2.imshow(
        "24h", video_frame
    )  # display the processed frame in a window named "My Face Detection Project"

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
