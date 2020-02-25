import numpy as np
import cv2
import pyzbar.pyzbar as pyzbar

 
image = cv2.imread('/home/diegosarina/Documentos/DIP/dip-qr-reader/documents/qr_images/4.jpeg')
#cv2.imshow("imagen",image)


barcodes = pyzbar.decode(image)

for barcode in barcodes:

    (x, y, w, h) = barcode.rect
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    barcodeData = barcode.data.decode("utf-8")
    barcodeType = barcode.type
    # draw the barcode data and barcode type on the image
    text = "{} ({})".format(barcodeData, barcodeType)
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
    0.5, (0, 0, 255), 2)
    # print the barcode type and data to the terminal
    print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))

# show the output image

cv2.imshow("Image", image)
cv2.waitKey(0)
