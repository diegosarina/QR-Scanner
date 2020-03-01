# Standard library imports

# Third party imports
import numpy as np
import cv2
import pyzbar.pyzbar as pyzbar

# Local application imports

# Constants variables
QR_NORTH = 0
QR_EAST = 1
QR_SOUTH = 2
QR_WEST = 3

OTSU_BINARIZATION = 0
THRES_BINARIZATION = 1

DEBUG_MODE_OFF = 0
DEBUG_MODE_ON = 1


class IntersPointError(Exception):
    pass

class OffLineError(Exception):
    pass


def distance(point_1, point_2):
    return np.linalg.norm(point_1 - point_2)


def distance_line_to_point(line_p1, line_p2, p):
    m = (line_p2[1] - line_p1[1]) / (line_p2[0] - line_p1[0])
    a = -m
    b = 1.0
    c = (m * line_p1[0]) - line_p1[1]
    dist = (a * p[0] + b * p[1] + c) / np.sqrt((a * a)+(b * b))

    return dist


def line_slope(point_1, point_2):
    dx = point_2[0] - point_1[0]
    dy = point_2[1] - point_1[1]
    if dy:
        return (dy / dx), 1
    
    return 0, 0


def update_corner(c1, ref, baseline, corner):
    temp_dist = distance(c1, ref)

    if(temp_dist > baseline):
        baseline = temp_dist
        corner = c1
        return baseline, corner
    
    return baseline, corner
    

def get_vertices(contours, c_id, slope):
    """
        A-----W-----B
        |           |
        Z           X
        |           |
        D-----Y-----C
    """
    box = cv2.boundingRect(contours[c_id])
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
    
    half_x = (A[0] + B[0]) / 2
    half_y = (A[1] + D[1]) / 2
    
    for i in range(len(contours[c_id])):

        if((contours[c_id][i][0][0] < half_x) and (contours[c_id][i][0][1] <= half_y)):
            dmax[2], M0 = update_corner(contours[c_id][i][0], C, dmax[2], M0)

        elif((contours[c_id][i][0][0] >= half_x) and (contours[c_id][i][0][1] < half_y)):
            dmax[3], M1 = update_corner(contours[c_id][i][0], D, dmax[3], M1)
        
        elif((contours[c_id][i][0][0] > half_x) and (contours[c_id][i][0][1] >= half_y)):
            dmax[0], M2 = update_corner(contours[c_id][i][0], A, dmax[0], M2)

        elif((contours[c_id][i][0][0] <= half_x) and (contours[c_id][i][0][1] > half_y)):
            dmax[1], M3 = update_corner(contours[c_id][i][0], B, dmax[1], M3)

    return [M0, M1, M2, M3]


def updateCornerOr(orientation, vertices):
    
    return {
        QR_NORTH: [vertices[0], vertices[1], vertices[2], vertices[3]],
        QR_EAST: [vertices[1], vertices[2], vertices[3], vertices[0]],
        QR_SOUTH: [vertices[2], vertices[3], vertices[0], vertices[1]],
        QR_WEST: [vertices[3], vertices[0],vertices[1],vertices[2]]
    }[orientation]


def intersection_between_two_lines(line1_p1, line1_p2, line2_p1, line2_p2):
    delta_x = (line1_p2[0] - line1_p1[0], line2_p2[0] - line2_p1[0])
    delta_y = (line1_p2[1] - line1_p1[1], line2_p2[1] - line2_p1[1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    determinant = det(delta_x, delta_y)
    if not determinant:
       raise IntersPointError('lines do not intersect')

    d = (det(line1_p2,line1_p1), det(line2_p2,line2_p1))
    x = float(det(d, delta_x)) / determinant
    y = float(det(d, delta_y)) / determinant
    
    return np.int32([x, y])


def image_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 220, (9, 9), L2gradient=True)
    return edges


def mass_center(contours):
    mass_centers = np.empty([len(contours),2])
    for i in range(len(contours)):
        moments = cv2.moments(contours[i])
        if moments["m00"]:
            mass_centers[i,:]= (
                moments["m10"] / moments["m00"],
                moments["m01"] / moments["m00"]
            )   
        else:
            mass_centers[i,:] = (0,0)
             
    return mass_centers


def search_candidate_markers(contours, hierarchy):
    marks_candidates = []

    for i in range(len(contours)):
        approx = cv2.approxPolyDP(
            contours[i],
            cv2.arcLength(contours[i],True)*0.02,
            True
        )
        if len(approx) == 4:
            k=i
            cc=0
            
            while (hierarchy[0,k,2] != -1):
                k = hierarchy[0,k,2]
                cc = cc+1
        
            if (hierarchy[0,k,2] != -1):
                cc = cc+1
        
            if cc>=5:
                marks_candidates.append(i) 
    
    return marks_candidates


def find_correct_mark(marks_candidates):
    A = marks_candidates[0]
    B = marks_candidates[1]
    C = marks_candidates[2]
    return A, B, C


def find_offline_point(mass_centers, A, B, C):
    AB = distance(mass_centers[A,:], mass_centers[B,:])
    BC = distance(mass_centers[B,:], mass_centers[C,:])
    CA = distance(mass_centers[C,:], mass_centers[A,:])

    if((AB > BC) and (AB > CA)):
        offline_point = C
        return offline_point, A, B
    elif ((CA > AB) and (CA > BC)):
        offline_point = B
        return offline_point, A, C
    elif((BC > AB) and (BC > CA)):
        offline_point = A
        return offline_point, B, C
    else:
        raise OffLineError


def get_orientation(align, slope, dist, mass_centers, mediana_1, mediana_2):
    if not align:
        botton = mediana_1
        rigth = mediana_2
    elif(slope < 0 and dist < 0):
        orientation= QR_NORTH
        if(mass_centers[mediana_1][1] > mass_centers[mediana_2][1]):
            botton = mediana_1
            rigth = mediana_2
        else:
            botton = mediana_2
            rigth = mediana_1       
    elif(slope>0 and dist<0):
        orientation= QR_EAST
        if(mass_centers[mediana_1][0] > mass_centers[mediana_2][0]):
            botton = mediana_2
            rigth = mediana_1
        else:
            botton = mediana_1
            rigth = mediana_2
    elif(slope<0 and dist>0):
        orientation= QR_SOUTH 
        if(mass_centers[mediana_1][1] > mass_centers[mediana_2][1]):
            botton = mediana_2
            rigth = mediana_1
        else:
            botton = mediana_1
            rigth = mediana_2
    elif(slope>0 and dist >0):
        orientation = QR_WEST
        if(mass_centers[mediana_1][0] > mass_centers[mediana_2][0]):
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
    
    return imagen
    

def binarizado_imagen(imagen, thresh=-1):

    if(thresh == -1):
        _, otsu = cv2.threshold(
            cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY),
            0,255,
            cv2.THRESH_BINARY+cv2.THRESH_OTSU
        )
        return otsu
    
    _, binary = cv2.threshold(
        cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY),
        thresh,255,
        cv2.THRESH_BINARY
    )
    return binary


def plot_marcadores(imagen, contornos, top, rigth, bottom, N, tickness=1):

    cv2.drawContours(imagen, contornos, top, (255, 200, 0), tickness)
    cv2.drawContours(imagen, contornos, rigth, (0, 0, 255), tickness)
    cv2.drawContours(imagen, contornos, bottom, (255, 0, 100), tickness)

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


def qr_scanner(image, thresh=-1, verbose=0):
    ret = ()
    bordes = image_edges(image)

    #APPROX SIMPLE -> SE QUEDA CON LOS PUNTOS MAS EXTERNOS DEL CONTORNO
    #APPROX NONE -> DEVUELVE TODOS LOS PUNTOS
    contornos, herencia = cv2.findContours(
        np.copy(bordes),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_TC89_KCOS
    )
    centros_de_masas = mass_center(contornos)
    marks_candidates = search_candidate_markers(contornos, herencia)
    
    if len(marks_candidates)>=3: #marcadores descubiertos

        A, B, C = find_correct_mark(marks_candidates)
        area_A = cv2.contourArea(contornos[A])
        area_B = cv2.contourArea(contornos[B])
        area_C = cv2.contourArea(contornos[C])

        area_promedio = (area_A + area_B + area_C) / 3

        if not ((area_A > area_promedio * 0.8 and area_A < area_promedio * 1.2)
            and (area_B > area_promedio * 0.8 and area_B < area_promedio * 1.2)
            and (area_C > area_promedio * 0.8 and area_C < area_promedio * 1.2)):
            return None
        
        try:
            top, mediana_1, mediana_2 = find_offline_point (centros_de_masas, A, B, C)
        except OffLineError:
            return None
        
        dist = distance_line_to_point(
            centros_de_masas[mediana_1],
            centros_de_masas[mediana_2],
            centros_de_masas[top]
        )
        slope, align = line_slope(
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
                        
            L = updateCornerOr(orientation,get_vertices(contornos, top, slope))

            M = updateCornerOr(orientation,get_vertices(contornos, rigth, slope))
            
            O = updateCornerOr(orientation,get_vertices(contornos, bottom, slope))
            try:
                N = intersection_between_two_lines(M[1], M[2], O[3], O[2]) #calculo del punto de interseccion N 
            except IntersPointError:
                return None
            
            try:
                pixel = image[N[1],N[0]]
            except:
                return None

            vert_externos = np.array(([L[0], M[1], N, O[3]]),dtype=np.float32)
            
            imagen_perspectiva_corregida = correccion_perspectiva(vert_externos,image)

            ret = (
                binarizado_imagen(
                    imagen_perspectiva_corregida,
                    thresh
                ),
                plot_marcadores(
                    np.copy(image), 
                    contornos,
                    top,
                    rigth,
                    bottom,
                    N,
                    tickness=3
                )
            ) 
           
            if(verbose):
                traces = np.zeros((np.shape(image)),dtype=np.uint8)
                cv2.drawContours(traces,contornos,top,(255,0,100),1)
                cv2.drawContours(traces,contornos,rigth,(255,0,100),1)
                cv2.drawContours(traces,contornos,bottom,(255,0,100),1)

                plot_vertices(traces, L, M, O, N)
                plot_lineas(traces, M, O, N)

                orientaciones = {
                    QR_NORTH : "NORTE",
                    QR_SOUTH : "SUR",
                    QR_EAST : "ESTE",
                    QR_WEST : "OSTE",
                }

                cv2.putText(
                    traces,  orientaciones[orientation],
                    (20,30), cv2.FONT_HERSHEY_PLAIN, 
                    1, (0,255,0), 1, 8
                )
                cv2.imshow("traces", traces)

    return ret


def qr_decoder(qr_codificado):
    decoded_data = None
    qrs = pyzbar.decode(qr_codificado)

    for qr in qrs:
        decoded_data = qr.data.decode("utf-8")
        
    return decoded_data

class ScannerComponent():
    def __init__(self, line_speed, percentage):
        self._percentage = percentage
        self._line_speed = line_speed
        self._iter = 0
    
    def set_input_frame(self, frame):
        middle_x = int(frame.shape[1] / 2)
        middle_y = int(frame.shape[0] / 2)

        delta_x = int((frame.shape[1] * self._percentage[0]) / 2)
        delta_y = int((frame.shape[0] * self._percentage[1]) / 2)

        self.min_x = middle_x - delta_x
        self.max_x = middle_x + delta_x
        self.min_y = middle_y - delta_y
        self.max_y = middle_y + delta_y
        
        self.frame = frame
        self._central_image = np.copy(
            frame[
                self.min_y:self.max_y,
                self.min_x:self.max_x
            ]
        )

    def get_central_image(self):
        return self._central_image

    def draw(self):
        background = cv2.GaussianBlur(self.frame, (25,25), 5)
        background[
            self.min_y:self.max_y,
            self.min_x:self.max_x
        ] = self._central_image

        rectangle_color = (169, 169, 172)
        line_color = (0, 0, 255)

        cv2.rectangle(
            background,
            (self.min_x, self.min_y),
            (self.max_x, self.max_y),
            rectangle_color,
            1
        )
        cv2.line(
            background,
            (self.min_x, self._iter),
            (self.max_x, self._iter),
            line_color,
            1
        )
        self._increase_iterator()
        return background

    def _increase_iterator(self):
        if(self._iter >= self.max_y or self._iter <= self.min_y):
            self._iter = self.min_y
        self._iter += self._line_speed