# Standard library imports
import pdb
import math
# Third party imports
import numpy as np
import cv2

# Local application imports
from functions import *

cap = cv2.VideoCapture(2)

it = 0

qr_viejo = ''

while(True):

    ret, img_original = cap.read()
    fondo, centro, p1, p2 = get_fondo_centro(img_original, 0.5, 0.5)
    centro_copia = np.copy(centro)
    fondo_decorado, it = decorador(fondo, it+3, p1, p2)
    cv2.imshow("QR DETECTOR", fondo_decorado)

    qr_decodificado = qr_decode(centro, DEBUG_MODE_OFF)
    
    if (qr_viejo != qr_decodificado) and (qr_decodificado):
        print(f"QR DATA: {qr_decodificado}")
        qr_viejo = qr_decodificado

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
