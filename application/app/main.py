import cv2
import numpy as np

print ("Hello World")
print ("OpenCV version",cv2.__version__)

img = cv2.imread('/home/diegosarina/Documentos/DIP/dip-qr-reader/documents/samples/image_samples/qr_type1.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('img',img)

ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

print(len(contours[0]))

cv2.drawContours(img, contours, -1, (0,255,0), 2)

cv2.imshow('contours',img)
cv2.waitKey(0)
cv2.destroyAllWindows()