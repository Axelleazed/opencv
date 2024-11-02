import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Angelina Jolie','Denzel Washington','Jennifer Lawrence','Johnny Depp','Will Smith',]

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
img = cv.imread(r'C:\Users\TECH\Desktop\opencv\photo\validation\13.jpeg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

faces_rect = haar_cascade.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in faces_rect :
    faces_roi = gray[y:y+h,x:x+w]

    label,confidece = face_recognizer.predict(faces_roi)
    print(f'LAbel ={people[label]} with a confidence of {confidece}')

    cv.putText(img ,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('detected',img)
cv.waitKey(0)

