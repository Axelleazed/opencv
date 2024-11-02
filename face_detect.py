import cv2 as cv

img = cv.imread(r'C:\Users\TECH\Desktop\opencv\photo\Angelina Jolie\001_fe3347c0.jpg')
cv.imshow('angelina',img)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')
face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=1)

print(f'Number of faces found is  = {len(face_rect)}')

for (x,y,w,h) in face_rect :
    cv.rectangle(img , (x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow('detecred faces',img)

cv.waitKey(0)
