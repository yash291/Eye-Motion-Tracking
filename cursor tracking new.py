import cv2
import numpy as np
import dlib

vid = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2) , int((p1.y + p2.y)/2)


while True:
    img, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        # x, y = face.left(), face.top()
        # x1, y1 = face.right(), face.bottom()
        # print(landmarks)

        landmarks = predictor(gray, face)
        #for left eye
        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)
        center_top = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom  = midpoint(landmarks.part(41), landmarks.part(40))

        hor_line = cv2.line(frame, left_point, right_point, (0,255,0), 2)
        ver_line = cv2.line(frame, center_top, center_bottom, (0,255,0),2)

        #for right eye
        left_point = (landmarks.part(42).x, landmarks.part(42).y)
        right_point = (landmarks.part(45).x, landmarks.part(45).y)
        center_top = midpoint(landmarks.part(43), landmarks.part(44))
        center_bottom  = midpoint(landmarks.part(47), landmarks.part(46))

        hor_line = cv2.line(frame, left_point, right_point, (0,255,0), 2)
        ver_line = cv2.line(frame, center_top, center_bottom, (0,255,0),2)
        
        
        # cv2.rectangle(frame, (x,y), (x1, y1),(0,255,0), 2)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
