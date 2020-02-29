# Standard library imports


# Third party imports
import numpy as np
import cv2
import pyzbar.pyzbar as pyzbar
# Local application imports
from functions import *


def video_mode(id_source_video = -1, debug_mode = DEBUG_MODE_OFF, binarition_mode = OTSU_BINARIZATION):

    cap = cv2.VideoCapture(id_source_video)

    it = 0

    qr_viejo = ''

    while(True):

        ret, img_original = cap.read()
        fondo, centro, p1, p2 = get_fondo_centro(img_original, 0.5, 0.5)
        centro_copia = np.copy(centro)
        fondo_decorado, it = decorador(fondo, it+3, p1, p2)
        cv2.imshow("QR DETECTOR", fondo_decorado)

        qr_decodificado = qr_decode(centro, debug_mode, binarition_mode)
        
        if (qr_viejo != qr_decodificado) and (qr_decodificado):
            print(f"QR DATA: {qr_decodificado}")
            qr_viejo = qr_decodificado

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def image_mode(path_to_image, debug_mode = DEBUG_MODE_OFF, binarition_mode = THRES_BINARIZATION):

    img = cv2.imread(path_to_image)
    print(path_to_image)
    print (img)
    while(True):
        cv2.imshow("QR DETECTOR", img)
        qr_decodificado = qr_decode(img, debug_mode, binarition_mode)
        print(f"QR DATA: {qr_decodificado}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
