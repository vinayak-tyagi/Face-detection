import cv2
import numpy as nm

print("PRESS ESC KEY TO EXIT")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray , 1.3,5)
    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w , y+h),(0,255,255),2)
        eye_gray = gray[y:y+h , x:x+w]
        eye_color = img[y:y+h , x:x+w]
        eyes = eye_cascade.detectMultiScale(eye_gray)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(eye_color,(ex , ey) , (ex+ew, ey+eh),(0,0,255), 2)


    cv2.imshow('img',img)
    num = cv2.waitKey(30) & 0xff
    if num == 27:
        break

cap.release()
cv2.destroyAllWindows() 
             
