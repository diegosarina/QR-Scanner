# Standard library imports
import pdb
import math
# Third party imports
import numpy as np
import cv2

# Local application imports
from functions import *

cap = cv2.VideoCapture(2)

top = None
bottom,rigth,orientation=None,None,None

it = 0

while(True):

    ret, img_original = cap.read()
    fondo, centro, p1, p2 = get_fondo_centro(img_original, 0.5, 0.5)
    centro_copia = np.copy(centro)
    fondo_decorado, it = decorador(fondo, it+3, p1, p2)
    cv2.imshow("QR DETECTOR", fondo_decorado)
    bordes = get_bordes(centro)

    img_2 = np.copy(img_original)

    #APPROX SIMPLE -> SE QUEDA CON LOS PUNTOS MAS EXTERNOS DEL CONTORNO
    #APPROX NONE -> DEVUELVE TODOS LOS PUNTOS
    contornos, herencia = cv2.findContours(np.copy(bordes),cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)

    centros_de_masas = get_centros_de_masas(contornos)
        
    mark, A, B, C = get_marcadores_indice(contornos, herencia)

    if mark>=3: #marcadores descubiertos
        area_A = cv2.contourArea(contornos[A])
        area_B = cv2.contourArea(contornos[B])
        area_C = cv2.contourArea(contornos[C])

        area_promedio = (area_A + area_B + area_C) / 3

        if not ((area_A > area_promedio * 0.8 and area_A < area_promedio * 1.2)
            and (area_B > area_promedio * 0.8 and area_B < area_promedio * 1.2)
            and (area_C > area_promedio * 0.8 and area_C < area_promedio * 1.2)):
            continue
        
        try:
            top, mediana_1, mediana_2 = get_offline(centros_de_masas, A, B, C)
        except OffLineExeption:
            continue
        
        dist = lineEquation(
            centros_de_masas[mediana_1],
            centros_de_masas[mediana_2],
            centros_de_masas[top]
        )
        slope, align = lineSlope(
            centros_de_masas[mediana_1],
            centros_de_masas[mediana_2]
        )

        orientation, bottom, rigth = get_orientation(
            align,
            slope,
            dist,
            centros_de_masas,
            mediana_1,
            mediana_2
        )

        if((cv2.contourArea(contornos[top]) > 10) 
            and (cv2.contourArea(contornos[rigth]) > 10) 
            and (cv2.contourArea(contornos[bottom]) > 10)):
                        
            L = updateCornerOr(orientation,getVerices(contornos, top, slope))

            M = updateCornerOr(orientation,getVerices(contornos, rigth, slope))
            
            O = updateCornerOr(orientation,getVerices(contornos, bottom, slope))
            try:
                N = getIntersectionPoint(M[1], M[2], O[3], O[2]) #calculo del punto de interseccion N 
            except IntersPointError:
                continue
            
            try:
                pixel = centro[N[1],N[0]]
            except:
                continue

            vert_externos = np.array(([L[0], M[1], N, O[3]]),dtype=np.float32)
            
            imagen_perspectiva_corregida = correccion_perspectiva(vert_externos,centro)

            qr_binarizado = binarizado_imagen(imagen_perspectiva_corregida)

            img_marcadores = plot_marcadores(
                np.copy(centro), 
                contornos,
                top,
                rigth,
                bottom,
                N
            )

            cv2.imshow("IMAGEN MARCADORES",img_marcadores)

            traces = np.zeros((np.shape(centro)),dtype=np.uint8)
            
            DBG = 0
            if(DBG == 1):
            
                cv2.drawContours(traces,contornos,top,(255,0,100),1)
                cv2.drawContours(traces,contornos,rigth,(255,0,100),1)
                cv2.drawContours(traces,contornos,bottom,(255,0,100),1)

                plot_vertices(traces, L, M, O, N)
                plot_lineas(traces, M, O, N)

                orientaciones = {
                    CV_QR_NORTH : "NORTE",
                    CV_QR_SOUTH : "SUR",
                    CV_QR_EAST : "ESTE",
                    CV_QR_WEST : "OSTE",
                }

                cv2.putText(
                    traces,  orientaciones[orientation],
                    (20,30), cv2.FONT_HERSHEY_PLAIN, 
                    1, (0,255,0), 1, 8
                )
                
                
                cv2.imshow("DBG", traces)
                cv2.imshow("Original",img_original)

            cv2.imshow("QR BINARIZADO", qr_binarizado)
            qr_Data, qr_Type = decode_qr(qr_binarizado)
            print(f"Tipo {qr_Type}, Data: {qr_Data}")


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    
