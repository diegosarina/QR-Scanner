# Standard library imports


# Third party imports
import numpy as np
import cv2
import pyzbar.pyzbar as pyzbar
# Local application imports


# Constants variables
CV_QR_NORTH = 0
CV_QR_EAST = 1
CV_QR_SOUTH = 2
CV_QR_WEST = 3

def distancia(c1, c2):

    delta_x = np.absolute(c1[0] - c2[0])
    delta_y = np.absolute(c1[1] - c2[1])

    return np.sqrt(delta_x * delta_x + delta_y * delta_y)


def lineEquation(c1, c2, c3):

    m = (c2[1] - c1[1]) / (c2[0] - c1[0])
    a = -m
    b = 1.0
    c = (m * c1[0]) - c1[1]
    pdist = (a * c3[0] + b * c3[1] + c) / np.sqrt((a * a)+(b * b))

    return pdist


def lineSlope(c1, c2):
    
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]

    if(dy != 0):
        
        return(dy / dx), 1
    
    return 0, 0


def updateCorner(c1, ref, baseline, corner):

    temp_dist = distancia(c1, ref)

    if(temp_dist > baseline):
        
        baseline = temp_dist
        corner = c1
        
        return baseline, corner
    
    return baseline, corner
    

def getVerices(contornos, c_id, slope):
    
    box = cv2.boundingRect(contornos[c_id])
    A = box[0:2]
    B = [A[0] + box[2], A[1]]
    C = [B[0], A[1] + box[3]]
    D = [A[0], C[1]]

    W = [(A[0] + B[0]) / 2, (A[1])]
    X = [B[0], (B[1] + C[1]) / 2]
    Y = [(C[0] + D[0]) / 2, C[1]]
    Z = [D[0], (D[1] + A[1]) / 2]

    dmax = np.array([0, 0, 0, 0])

    M0, M1, M2, M3 = [0,0], [0,0], [0,0], [0,0]

    if(slope > 5 or slope < -5):
        
        for i in range(len(contornos[c_id])):
            pd1 = lineEquation(C, A, contornos[c_id][i][0])
            pd2 = lineEquation(B, D, contornos[c_id][i][0])
        
            if((pd1 >= 0.0) and (pd2 > 0.0)):
                dmax[1], M1 = updateCorner(contornos[c_id][i][0], W, dmax[1], M1)
            
            elif ((pd1 > 0.0) and (pd2 <= 0.0)):
                dmax[2], M2 = updateCorner(contornos[c_id][i][0], X, dmax[2], M2)

            elif ((pd1 <= 0.0) and (pd2 < 0.0)):
                dmax[3], M3 = updateCorner(contornos[c_id][i][0], Y, dmax[3], M3)
            
            elif ((pd1 < 0.0) and (pd2 >= 0.0)):
                dmax[0], M0 = updateCorner(contornos[c_id][i][0], Z, dmax[0], M0)

    else:
        half_x = (A[0] + B[0]) / 2
        half_y = (A[1] + D[1]) / 2
        
        for i in range(len(contornos[c_id])):

            if((contornos[c_id][i][0][0] < half_x) and (contornos[c_id][i][0][1] <= half_y)):
                dmax[2], M0 = updateCorner(contornos[c_id][i][0], C, dmax[2], M0)

            elif((contornos[c_id][i][0][0] >= half_x) and (contornos[c_id][i][0][1] < half_y)):
                dmax[3], M1 = updateCorner(contornos[c_id][i][0], D, dmax[3], M1)
            
            elif((contornos[c_id][i][0][0] > half_x) and (contornos[c_id][i][0][1] >= half_y)):
                dmax[0], M2 = updateCorner(contornos[c_id][i][0], A, dmax[0], M2)

            elif((contornos[c_id][i][0][0] <= half_x) and (contornos[c_id][i][0][1] > half_y)):
                dmax[1], M3 = updateCorner(contornos[c_id][i][0], B, dmax[1], M3)

    return [M0, M1, M2, M3]


def updateCornerOr(orientation, verticesIN):
    
    return {
        CV_QR_NORTH: [verticesIN[0], verticesIN[1], verticesIN[2], verticesIN[3]],
        CV_QR_EAST: [verticesIN[1], verticesIN[2], verticesIN[3], verticesIN[0]],
        CV_QR_SOUTH: [verticesIN[2], verticesIN[3], verticesIN[0], verticesIN[1]],
        CV_QR_WEST: [verticesIN[3], verticesIN[0],verticesIN[1],verticesIN[2]]
    }[orientation]


class IntersPointError(Exception):
    pass

def getIntersectionPoint(a1,a2,b1,b2):

    delta_a = a2 - a1
    delta_b = b2 - b1

    m_a = delta_a[1] / delta_a[0]
    m_b = delta_b[1] / delta_b[0]

    M = np.array([[-m_a,1],[-m_b,1]])

    try:
        inversa_M = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        raise IntersPointError
    
    b_a = a1[1] - (m_a) * a1[0]
    b_b = b1[1] - (m_b) * b1[0]

    B = np.array([[b_a],[b_b]])

    intersection_point = np.dot(inversa_M,B)

    return np.int32(np.transpose(intersection_point))[0]



def get_fondo_centro(frame, porcent_x, porcent_y):
    
    centro_x = int(frame.shape[1] / 2)
    centro_y = int(frame.shape[0] / 2)

    delta_x = int((frame.shape[1] * porcent_x) / 2)
    delta_y = int((frame.shape[0] * porcent_y) / 2) 


    fondo = np.copy(frame)
    fondo = cv2.GaussianBlur(fondo,(25,25),5)
    
    centro = np.copy(frame[centro_y - delta_y : centro_y + delta_y, centro_x - delta_x : centro_x + delta_x])
    fondo[centro_y - delta_y : centro_y + delta_y, centro_x - delta_x : centro_x + delta_x] = centro
    
    p1 = (centro_x - delta_x, centro_y - delta_y)
    p2 = (centro_x + delta_x, centro_y + delta_y)

    return fondo, centro, p1, p2


def decorador(frame, iterator, p1, p2):
    #p1 -> is the top left point
    #p2 -> is the bottom rigth point

    rectangle_color = (169, 169, 172)
    line_color = (0, 0, 255)

    if iterator > p2[1]:
        iterator = p1[1]
    
    if iterator < p1[1]:
        iterator = p1[1]

    cv2.rectangle(frame, p1, p2, rectangle_color, 1)
    cv2.line(frame, (p1[0], iterator),(p2[0] , iterator), line_color, 1)

    return frame, iterator


def get_bordes(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gray, 20, 220, (9, 9), L2gradient=True)

    return bordes


def get_centros_de_masas(contornos):

    centro_masa = np.empty([len(contornos),2])

    for i in range(len(contornos)):
        
        momentos = cv2.moments(contornos[i])
        if momentos["m00"]:
            centro_masa[i,:]= (
                momentos["m10"] / momentos["m00"],
                momentos["m01"] / momentos["m00"]
            )   
        else:
            centro_masa[i,:] = (0,0)
             
    return centro_masa


def get_marcadores_indice(contornos, herencia):

    mark = 0
    A, B, C = None, None, None
    for i in range(len(contornos)):
        
        approx = cv2.approxPolyDP(contornos[i],cv2.arcLength(contornos[i],True)*0.02,True)
        if len(approx) == 4:
            k=i
            cc=0
            
            while (herencia[0,k,2] != -1):
                k = herencia[0,k,2]
                cc = cc+1
        
            if (herencia[0,k,2] != -1):
                cc = cc+1
        
            if cc>=5: 
                if(mark==0):
                    A = i
                elif (mark==1):
                    B=i
                elif (mark==2):
                    C=i
                mark = mark+1
    
    return mark, A, B, C


def get_offline(centros_de_masas, A, B, C):
    
    AB = distancia(centros_de_masas[A,:], centros_de_masas[B,:])
    BC = distancia(centros_de_masas[B,:], centros_de_masas[C,:])
    CA = distancia(centros_de_masas[C,:], centros_de_masas[A,:])

    if((AB>BC)and(AB>CA)):
        valor_mayor = C
        mediana_1 = A
        mediana_2 = B
    elif ((CA>AB)and(CA>BC)):
        valor_mayor = B
        mediana_1 = A
        mediana_2 = C
    elif((BC > AB) and (BC > CA)):
        valor_mayor = A
        mediana_1 = B
        mediana_2 = C
    
    return valor_mayor, mediana_1, mediana_2


def get_orientation(align, slope, dist, centros_de_masas, mediana_1, mediana_2):

    if not align:
        botton = mediana_1
        rigth = mediana_2
    elif(slope < 0 and dist < 0): #orientacion norte
        orientation= CV_QR_NORTH
        if(centros_de_masas[mediana_1][1] > centros_de_masas[mediana_2][1]):
            botton = mediana_1
            rigth = mediana_2
        else:
            botton = mediana_2
            rigth = mediana_1       
    elif(slope>0 and dist<0): #orientacion este
        orientation= CV_QR_EAST
        if(centros_de_masas[mediana_1][0] > centros_de_masas[mediana_2][0]):
            botton = mediana_2
            rigth = mediana_1
        else:
            botton = mediana_1
            rigth = mediana_2
    elif(slope<0 and dist>0): #orientacion sur
        orientation= CV_QR_SOUTH 
        if(centros_de_masas[mediana_1][1] > centros_de_masas[mediana_2][1]):
            botton = mediana_2
            rigth = mediana_1
        else:
            botton = mediana_1
            rigth = mediana_2
    elif(slope>0 and dist >0):#orientacion oeste
        orientation = CV_QR_WEST
        if(centros_de_masas[mediana_1][0] > centros_de_masas[mediana_2][0]):
            botton = mediana_1
            rigth = mediana_2
        else:
            botton = mediana_2
            rigth = mediana_1    

    return orientation, botton, rigth


def correccion_perspectiva(vertices, imagen):

    dst = np.array(((40,40),(440,40),(440,440),(40,440)),dtype=np.float32)
    warp_matrix = cv2.getPerspectiveTransform(vertices,dst)
    warped = cv2.warpPerspective(imagen, warp_matrix,(500,500),borderMode=cv2.BORDER_REPLICATE)
    imagen = cv2.copyMakeBorder(warped,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255))
    
    _, binary = cv2.threshold(cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY), 100,255,cv2.THRESH_BINARY)
    _, otsu = cv2.threshold(cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return binary, otsu

def plot_marcadores(imagen, contornos, top, rigth, bottom, N, tickness=1):

    cv2.drawContours(imagen, contornos, top, (255, 200, 0), tickness)
    cv2.drawContours(imagen, contornos, rigth, (0, 0, 255), tickness)
    cv2.drawContours(imagen, contornos, bottom, (255, 0, 100), tickness)
    cv2.circle(imagen, tuple(N), 5, (0, 0, 255), 5, tickness)

    return imagen


def plot_vertices(imagen, L, M, O, N):

    cv2.circle(imagen, tuple(L[0]), 2, (255, 255, 0), -1, 8, 0)
    cv2.circle(imagen, tuple(L[1]), 2, (0, 255, 0), -1, 8, 0)
    cv2.circle(imagen, tuple(L[2]), 2, (0, 0, 255), -1, 8, 0)
    cv2.circle(imagen, tuple(L[3]), 2, (128, 128, 128), -1, 8, 0)

    cv2.circle(imagen, tuple(M[0]), 2, (255, 255, 0), -1, 8, 0)
    cv2.circle(imagen, tuple(M[1]), 2, (0, 255, 0), -1, 8, 0)
    cv2.circle(imagen, tuple(M[2]), 2, (0, 0, 255), -1, 8, 0)
    cv2.circle(imagen, tuple(M[3]), 2, (128, 128, 128), -1, 8, 0)

    cv2.circle(imagen, tuple(O[0]), 2, (255, 255, 0), -1, 8, 0)
    cv2.circle(imagen, tuple(O[1]), 2, (0, 255, 0), -1, 8, 0)
    cv2.circle(imagen, tuple(O[2]), 2, (0, 0, 255), -1, 8, 0)
    cv2.circle(imagen, tuple(O[3]), 2, (128, 128, 128), -1, 8, 0)

    cv2.circle(imagen, tuple(N), 5, (70, 252, 252), -1, 8, 0)


def plot_lineas(imagen, M, O, N):
    
    cv2.line(imagen, tuple(M[1]), tuple(N), (0, 0, 255), 1, 8, 0)
    cv2.line(imagen, tuple(O[3]), tuple(N), (0, 0, 255), 1, 8, 0)


def decode_qr(qr_codificado):

    qr_Data, qr_Type = None, None
    qrs = pyzbar.decode(qr_codificado)
    for qr in qrs:

        qr_Data = qr.data.decode("utf-8")
        qr_Type = qr.type

        
    return qr_Data, qr_Type