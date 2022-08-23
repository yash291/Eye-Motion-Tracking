# import the opencv library
import cv2
import numpy as np

# define a video capture object
vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(r"C:\Users\HP\Desktop\Eye Motion Tracking\haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier(r"C:\Users\HP\Desktop\Eye Motion Tracking\haarcascade_eye.xml")

while True:
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#cvt -> convertcolour
    faces = face_cascade.detectMultiScale(gray,2.3,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.putText(frame,'Face',(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(250,250,250),2)
    
    eyes = eyes_cascade.detectMultiScale(gray,2.3,4)

    for (x1,y1,w1,h1) in eyes:
        cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(255,0,0),3)
        cv2.putText(frame,'Eyes',(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(250,250,250),2)
    



    # roi = frame[269:795, 537:1416]
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
