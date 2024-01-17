from ultralytics import YOLO
import cv2
import math
import socket

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# VARS
detect = 'face' # person or face
confidence_threshold = 0.7 # thresold
distance_threshold = 50
scale_percent = 150 # image size

# models
model = YOLO("faceModel.pt")

# object classes
classNames = ["face"]

def getPos():
    hex_data = '010000070000000081090612FF'
    data = bytes.fromhex(hex_data)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.sendto(data, ("187.101.100.107", 52381))

    udp_socket.settimeout(5.0)
    response, _ = udp_socket.recvfrom(1024)

    pan_val = response.hex()[18:26]
    pan_val_text = int(pan_val[1]+pan_val[3]+pan_val[5]+pan_val[7], 16)

    tilt_val = response.hex()[26:34]
    tilt_val_text = int(tilt_val[1]+tilt_val[3]+tilt_val[5]+tilt_val[7], 16)

    udp_socket.close()

    return [hex(pan_val_text), hex(tilt_val_text)]

def moveCamera(pOrientation, speed):
    if pOrientation == 'up':
        hex_data = '0100000700000000810106010'+speed+'0301FF'
        data = bytes.fromhex(hex_data)
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.sendto(data, ("187.101.100.107", 52381))
        udp_socket.close()
    elif pOrientation == 'down':
        hex_data = '0100000700000000810106010'+speed+'0302FF'
        data = bytes.fromhex(hex_data)
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.sendto(data, ("187.101.100.107", 52381))
        udp_socket.close()
    elif pOrientation == 'right':
        hex_data = '0100000700000000810106010'+speed+'0103FF'
        data = bytes.fromhex(hex_data)
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.sendto(data, ("187.101.100.107", 52381))
        udp_socket.close()
    elif pOrientation == 'left':
        hex_data = '0100000700000000810106010'+speed+'0203FF'
        data = bytes.fromhex(hex_data)
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.sendto(data, ("187.101.100.107", 52381))
        udp_socket.close()
    elif pOrientation == 'upRight':
        hex_data = '0100000700000000810106010'+speed+'0102FF'
        data = bytes.fromhex(hex_data)
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.sendto(data, ("187.101.100.107", 52381))
        udp_socket.close()
    elif pOrientation == 'upLeft':
        hex_data = '0100000700000000810106010'+speed+'0202FF'
        data = bytes.fromhex(hex_data)
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.sendto(data, ("187.101.100.107", 52381))
        udp_socket.close()
    elif pOrientation == 'downRight':
        hex_data = '0100000700000000810106010'+speed+'0101FF'
        data = bytes.fromhex(hex_data)
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.sendto(data, ("187.101.100.107", 52381))
        udp_socket.close()
    elif pOrientation == 'downLeft':
        hex_data = '0100000700000000810106010'+speed+'0201FF'
        data = bytes.fromhex(hex_data)
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.sendto(data, ("187.101.100.107", 52381))
        udp_socket.close()
    elif pOrientation == 'stop':
        hex_data = '01000007000000008101060100000303FF'
        data = bytes.fromhex(hex_data)
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.sendto(data, ("187.101.100.107", 52381))
        udp_socket.close()   

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    stopCamera = True
    
    # coordinates
    for r in results:
        print(getPos())
        boxes = r.boxes

        for box in boxes:
            confidence = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            if confidence < confidence_threshold or classNames[cls] != classNames[0]:
                continue
            
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            targetX, targetY, centerX, centerY = int((x1+x2)/2), int((y1+y2)/2), int(img.shape[1]/2), int(img.shape[0]/2)
            distance = math.sqrt((targetX - centerX)**2 + (targetY - centerY)**2)

            if distance > distance_threshold:
                speed = '505'
                if distance > distance_threshold*2:
                    speed = 'F0F'
                elif distance > distance_threshold*1.75:
                    speed = 'C0C'
                elif distance > distance_threshold*1.5:
                    speed = 'B0B'
                elif distance > distance_threshold*1.25:
                    speed = '808'
                
                if targetX < centerX and targetY < centerY:
                    stopCamera = False
                    moveCamera(pOrientation='downRight', speed=speed)
                elif targetX > centerX and targetY < centerY:
                    stopCamera = False
                    moveCamera(pOrientation='downLeft', speed=speed)
                elif targetX < centerX and targetY > centerY:
                    stopCamera = False
                    moveCamera(pOrientation='upRight', speed=speed)
                elif targetX > centerX and targetY > centerY:
                    stopCamera = False
                    moveCamera(pOrientation='upLeft', speed=speed)
               
            point_coordinates = (targetX, targetY)
            point_color = (255, 165, 0)
            point_thickness = -1
            cv2.circle(img, point_coordinates, 2, point_color, point_thickness)

            point_coordinates = (centerX, centerY)
            cv2.circle(img, point_coordinates, 2, point_color, point_thickness)

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 165, 0), 3)

            # object details
            org = [x1, y1-10]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.7
            color = (255, 165, 0)
            thickness = 2

            cv2.putText(img, classNames[cls] + ' ' + str((confidence*100)) + '%', org, font, fontScale, color, thickness)
            break

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    if stopCamera == True:
        moveCamera(pOrientation='stop', speed='000')

    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        moveCamera(pOrientation='stop')
        break

cap.release()
cv2.destroyAllWindows()