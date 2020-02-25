import cv2
import numpy as np
import pdb
import math
import copy
import pyzbar.pyzbar as pyzbar
def distancia(c1,c2):
    
    return np.sqrt(math.pow(np.absolute(c1[0] - c2[0]),2)+math.pow(np.absolute(c1[1] - c2[1]),2))

def lineEquation(c1,c2,c3):

    m = (c2[1]-c1[1])/(c2[0]-c1[0])
    a = -m
    b = 1.0
    c = (m * c1[0]) - c1[1]

    pdist = (a*c3[0]+b*c3[1]+c)/np.sqrt((a*a)+(b*b))

    return pdist

def lineSlope(c1,c2):
    
    dx=c2[0]-c1[0]
    dy=c2[1]-c1[1]

    if(dy!=0):
        return(dy/dx),1
    else:
        return 0,0

def updateCorner(c1,ref, baseline, corner):

    temp_dist = distancia(c1,ref)

    if(temp_dist > baseline):
        baseline = temp_dist
        corner = c1
        return baseline, corner
    
    return baseline, corner
    

def getVerices(contornos, c_id, slope):
    
    box = cv2.boundingRect(contornos[c_id])
    A = box[0:2]
    B = [A[0]+box[2],A[1]]
    C = [B[0],A[1]+box[3]]
    D = [A[0],C[1]]

    W = [(A[0]+B[0])/2,(A[1])]
    X = [B[0],(B[1]+C[1])/2]
    Y = [(C[0]+D[0])/2, C[1]]
    Z = [D[0],(D[1]+A[1])/2]

    dmax = np.array([0,0,0,0])

    M0,M1,M2,M3 = 0,0,0,0

    if(slope>5 or slope<-5):
        
        for i in range(len(contornos[c_id])):
            pd1 = lineEquation(C,A,contornos[c_id][i])
            pd2 = lineEquation(B,D,contornos[c_id][i])
        
            if((pd1 >= 0.0) and (pd2 > 0.0)):
                dmax[1], M1 = updateCorner(contornos[c_id][i][0],W,dmax[1], M1)
            
            elif ((pd1 > 0.0) and (pd2 <= 0.0)):
                dmax[2], M2 = updateCorner(contornos[c_id][i][0],X,dmax[2], M2)

            elif ((pd1 <= 0.0) and (pd2 < 0.0)):
                dmax[3], M3 = updateCorner(contornos[c_id][i][0],Y,dmax[3], M3)
            
            elif ((pd1 < 0.0) and (pd2 >= 0.0)):
                dmax[0], M0 = updateCorner(contornos[c_id][i][0],Z,dmax[0], M0)

            else:
                pass

            i = i+1
    else:
        half_x = (A[0]+B[0])/2
        half_y = (A[1]+D[1])/2
        #pdb.set_trace()
        
        for i in range(len(contornos[c_id])):
            #REVISAR! 
            if((contornos[c_id][i][0][0] < half_x) and (contornos[c_id][i][0][1] <= half_y)):
                dmax[2], M0 = updateCorner(contornos[c_id][i][0],C,dmax[2],M0)

            elif((contornos[c_id][i][0][0] >= half_x) and (contornos[c_id][i][0][1] < half_y)):
                dmax[3], M1 = updateCorner(contornos[c_id][i][0],D,dmax[3],M1)
            
            elif((contornos[c_id][i][0][0] > half_x) and (contornos[c_id][i][0][1] >= half_y)):
                dmax[0], M2 = updateCorner(contornos[c_id][i][0],A,dmax[0],M2)

            elif((contornos[c_id][i][0][0] <= half_x) and (contornos[c_id][i][0][1] > half_y)):
                dmax[1], M3 = updateCorner(contornos[c_id][i][0],B,dmax[1],M3)

    return [M0, M1, M2, M3]

def updateCornerOr(orientation, verticesIN):
    
    if (orientation == CV_QR_NORTH):
        
        return [verticesIN[0], verticesIN[1],verticesIN[2],verticesIN[3]]
    
    elif (orientation == CV_QR_EAST):

        return [verticesIN[1], verticesIN[2],verticesIN[3],verticesIN[0]]

    elif (orientation == CV_QR_SOUTH):

        return [verticesIN[2], verticesIN[3],verticesIN[0],verticesIN[1]]

    elif (orientation == CV_QR_WEST):

        return [verticesIN[3], verticesIN[0],verticesIN[1],verticesIN[2]]

def cross(v1,v2):

    return v1[0]*v2[1] - v1[1]*v2[0]

def getIntersectionPoint(a1, a2, b1, b2):

    p = a1
    q = b1 
    r = (a2-a1)
    s = (b2-b1)

    if (cross(r,s)== 0):
        return int(0)
    
    t = cross(q-p,s)/cross(r,s)
    intersection = p + t*r
    return np.array((int(intersection[0]),int(intersection[1])),dtype=np.int32)

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged


img_original = cv2.imread('/home/diegosarina/Documentos/DIP/dip-qr-reader/documents/qr_images/4.jpeg')
#img_2 = copy.copy(img_original)
img_2 = np.copy(img_original)
img_para_final = np.copy(img_original)
img_final_gris = cv2.cvtColor(img_para_final, cv2.COLOR_BGR2GRAY)
A = None
B = None
C = None
top = None
botton,rigth,orientation=None,None,None
CV_QR_NORTH = 0
CV_QR_EAST = 1
CV_QR_SOUTH = 2
CV_QR_WEST = 3
#img_2 = cv2.adaptiveThreshold(cv2.cvtColor(img_original,cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

#img_original = cv2.GaussianBlur(img_original,(3,3),0)

#th3 = cv2.adaptiveThreshold(fil,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#            cv2.THRESH_BINARY,51,11)


gray = cv2.cvtColor(img_original,cv2.COLOR_BGR2GRAY)
#ret, otsu = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#bordes = auto_canny(otsu)
bordes = cv2.Canny(gray,20,220,(9,9),L2gradient=True)
#bordes = cv2.dilate(bordes,(9,9))
#bordes = cv2.morphologyEx(bordes,cv2.MORPH_CLOSE,(9,9))
#cv2.imshow("bordes", bordes)
#cv2.imshow("gris", gray)
#APPROX SIMPLE -> SE QUEDA CON LOS PUNTOS MAS EXTERNOS DEL CONTORNO
#APPROX NONE -> DEVUELVE TODOS LOS PUNTOS
contornos, herencia = cv2.findContours(bordes,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)

#cv2.drawContours(img_original,contornos,-1,(0,255,0),1)
centro_masa = np.empty([len(contornos),2])

for i in range(len(contornos)):
    
    momentos = cv2.moments(contornos[i])
    if momentos["m00"] == 0:
        centro_masa[i,:] = (0,0)
    else:
        centro_masa[i,:]= (momentos["m10"]/momentos["m00"],
                        momentos["m01"]/momentos["m00"])    

    img_original = cv2.circle(img_2,(int(centro_masa[i,0]),int(centro_masa[i,1])),1,(0,0,255),1)                      
    
#pdb.set_trace()
#cv2.imshow("canny",img_original)
mark=0
for i in range(len(contornos)):
    k=i
    cc=0

    while (herencia[0,k,2] != -1):
        k = herencia[0,k,2]
        cc = cc+1
    
    if (herencia[0,k,2] != -1):
        cc = cc+1
    
    if cc>=5:
        img_original = cv2.putText(img_original,str(i),(int(centro_masa[i,0]),int(centro_masa[i,1])),
            cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)  
        if(mark==0):
            A = i
        elif (mark==1):
            B=i
        elif (mark==2):
            C=i
        mark = mark+1

if mark>=3: #marcadores descubiertos
    AB = distancia(centro_masa[A,:],centro_masa[B,:])
    BC = distancia(centro_masa[B,:],centro_masa[C,:])
    CA = distancia(centro_masa[C,:],centro_masa[A,:])

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
    
    top = valor_mayor
    
    dist = lineEquation(centro_masa[mediana_1],centro_masa[mediana_2],centro_masa[valor_mayor])
    slope,align = lineSlope(centro_masa[mediana_1],centro_masa[mediana_2])

    if align==0:
        botton = mediana_1
        rigth = mediana_2
    elif(slope<0 and dist<0): #orientacion norte
        orientation= CV_QR_NORTH

        if(centro_masa[mediana_1][1] > centro_masa[mediana_2][1]):
            botton = mediana_1
            rigth = mediana_2
        else:
            botton = mediana_2
            rigth = mediana_1
        
    elif(slope>0 and dist<0): #orientacion este
        orientation= CV_QR_EAST
        
        if(centro_masa[mediana_1][0] > centro_masa[mediana_2][0]):
            botton = mediana_2
            rigth = mediana_1
        else:
            botton = mediana_1
            rigth = mediana_2

    elif(slope<0 and dist>0): #orientacion sur
        orientation= CV_QR_SOUTH 

        if(centro_masa[mediana_1][1] > centro_masa[mediana_2][1]):
            botton = mediana_2
            rigth = mediana_1
        else:
            botton = mediana_1
            rigth = mediana_2
               
    elif(slope>0 and dist >0):#orientacion oeste
        orientation = CV_QR_WEST
        
        if(centro_masa[mediana_1][0] > centro_masa[mediana_2][0]):
            botton = mediana_1
            rigth = mediana_2
        else:
            botton = mediana_2
            rigth = mediana_1

    if( (cv2.contourArea(contornos[top]) > 10) and 
        (cv2.contourArea(contornos[rigth]) > 10) and 
        (cv2.contourArea(contornos[botton]) > 10)):
        
        verticesL = getVerices(contornos,top,slope)
        verticesM = getVerices(contornos,rigth,slope)
        verticesO = getVerices(contornos,botton,slope)
        
        L = updateCornerOr(orientation,verticesL)
        M = updateCornerOr(orientation,verticesM)
        O = updateCornerOr(orientation,verticesO)

        N = getIntersectionPoint(M[1],M[2],O[3],O[2]) #CALCULO DE INTERSECCION PUNTO N

        vert_externos = np.array(([L[0],M[1],N,O[3]]),dtype=np.float32) #en el codigo SRC
        
        dst = np.array(((0,0),(400,0),(400,400),(0,400)),dtype=np.float32)
        #pdb.set_trace()
        bordes_color = cv2.cvtColor(bordes,cv2.COLOR_GRAY2BGR)
        
        bordes_color[int(vert_externos[0,1]),int(vert_externos[0,0])] = (0,255,0)
        bordes_color[int(vert_externos[1,1]),int(vert_externos[1,0])] = (0,255,0)
        bordes_color[int(vert_externos[2,1]),int(vert_externos[2,0])] = (0,255,0)
        bordes_color[int(vert_externos[3,1]),int(vert_externos[3,0])] = (0,255,0)
        bordes_color[int(M[2][1]),int(M[2][0])] = (0,255,0)
        bordes_color[int(O[2][1]),int(O[2][0])] = (0,255,0)
        cv2.imshow("bordes color", bordes_color )
        #cv2.circle(img_2,(vert_externos[0,0],vert_externos[0,1]),5,(0,0,255),5)
        cv2.imshow("IMAGEN PIXEL", img_2)
        #cv2.circle(img_2,(vert_externos[1,0],vert_externos[1,1]),5,(0,0,255),5)
        #cv2.circle(img_2,(vert_externos[2,0],vert_externos[2,1]),5,(0,0,255),5)
        #cv2.circle(img_2,(vert_externos[3,0],vert_externos[3,1]),5,(0,0,255),5)

        warp_matrix = cv2.getPerspectiveTransform(vert_externos,dst)

        warped = cv2.warpPerspective(img_final_gris, warp_matrix,(400,400))

        imagen = cv2.copyMakeBorder(warped,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255))
        
        #img_gris = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)

        ret, thress = cv2.threshold(imagen, 100,255,cv2.THRESH_BINARY)

        cv2.drawContours(img_2,contornos,top,(255,200,0))
        cv2.drawContours(img_2,contornos,rigth,(0,0,255))
        cv2.drawContours(img_2,contornos,botton,(255,0,100))
        #print(len(contornos[746])) 
        cv2.circle(img_2,(N[0],N[1]),5,(0,0,255),5)
        #qr_final = cv2.threshold(img_gris, 127, 255, cv2.THRESH_BINARY)
        
        ret, qr_final = cv2.threshold(imagen,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
        cv2.imshow("IMG ORIGINAL",img_original)
        cv2.imshow("QR THRES",thress )
        cv2.imshow("QR OTSU", qr_final)
        cv2.imshow("IMGAGEN DEBUG ",img_2)
        cv2.imshow("QR", imagen)

        barcodes = pyzbar.decode(thress)
        img_barcode = np.copy(thress)
        for barcode in barcodes:

            (x, y, w, h) = barcode.rect
            cv2.rectangle(img_barcode, (x, y), (x + w, y + h), (0, 0, 255), 2)
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
            # draw the barcode data and barcode type on the image
            text = "{} ({})".format(barcodeData, barcodeType)
            cv2.putText(img_barcode, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, (0, 0, 255), 2)
            # print the barcode type and data to the terminal
            print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))

        cv2.imshow("barcode",img_barcode)
        cv2.waitKey(0)

#pdb.set_trace()
#cv2.imshow("sd",img_original)
