import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("images/car-man.png")

gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)

cv2.imshow("identified Faces", img)
cv2.waitKey(0) #processed image will close as soon as the program is run. Wait key will add a delay to prevent this
cv2.destroyAllWindows() # close all existing windows




