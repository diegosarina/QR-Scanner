import numpy as np
import cv2 as cv2
import scipy.signal as sp
import scipy
import matplotlib.pylab as plt
import os
from sklearn.cluster import KMeans
from pandas import DataFrame
from scipy.spatial.distance import pdist
import pdb
# PATH TO IMAGES
dirname = os.path.abspath('./documents/qr_images/real_qr/') 
image= os.path.join(dirname,'qr_type1_north.jpg')
# FUNCTION

def im_binary(img):
    th1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,5)
    return th1

def im_resize(img,scale_percent,interpolation_type):
    width = int(img.shape[1]*scale_percent/100)
    height = int(img.shape[0]*scale_percent/100)
    dim = (width,height)
    ret = cv2.resize(img,dim,interpolation=interpolation_type)
    return ret

def mod_Tr_Fn (img):
    # If the vector in is [0 0 1 0 1 1 1 0 0], out = [2 1 1 3 2].
    im_out = [np.ones]
    im_out = np.ones((img.shape[0]),dtype=int)
    #print(img)
    k = 0

    for i in range(0,len(img)-1,1):
        if img[i] == img[i+1]:
            im_out[k] = im_out[k] + 1
        else:
            k = k + 1
        #print(f"im out{im_out}")
    im_out =  im_out[:k+1]
    print(f'resultado final {im_out}')
    return im_out

def checkRatio_Fn(A,Pat):
    # total size of all the pixels in the sequence we looking at
    size = np.sum(A)
    out = None
    # size of the modules, ex 7 for the FIP-finder
    pat_size = np.sum(Pat)

    # if smaller than pattern size, no patter found
    if(size<pat_size):
        out = False
    
    # Calculate the size of one module
    cor_coeff = 0.4 # default is 0.4
    module_size = size/pat_size

    # how much the pattern can vary and still be acceptable
    maxV = module_size * cor_coeff

    # determine if the vector is good enough
    if np.all((np.absolute(np.multiply(module_size,Pat)-A) <= maxV)):
        out = True
    else:
        out = False
    
    return out, module_size

def find_Probable_FIPS_fn(img):
    pattern = np.array((1,1,3,1,1))
    number_of_fips = 0
    ################convertir location_FIPS a lista
    location_FIPS = []
    #location_FIPS.append([row,col])
    #location_FIPS = [0,0]
    width = int(img.shape[1])
    height = int(img.shape[0])
    dim = (width,height)

    for row in range(0,width,1):
        tr_line = img[row,:] #pick out the entire row
        
        # calculate the appearence of the modules in row.
        length_modules = mod_Tr_Fn(tr_line)

        pixel_pos_col = 1 # check sequence to find approbriate pattern similar to finder patterns

        for i in range (0,len(length_modules)-4,1):
            vectorFIP = length_modules[i:i+5]
            (isFIP,moduleSize) = checkRatio_Fn(vectorFIP,pattern)

            # The ratio is correct and the first value is black, we should search in that column
            if (isFIP and (img[row,pixel_pos_col] == 0)):
                col = int(pixel_pos_col + np.floor(np.sum(vectorFIP)/2)) # Specific column for search

                pixel_pos_row = 1 # Punto inicial
                
                # Busca a traves de todas las columnas y ve si el patron FIP es encontrado
                # si lo es, lo almacena como candidato
                tr_col = mod_Tr_Fn(img[:,col])

                for j in range(0,len(tr_col)-4,1):
                    vector_row_FIP = tr_col[j:j+5]
                    rowFIP, _ = checkRatio_Fn(vector_row_FIP, pattern)

                    # One more time find the center position of the finder pattern in the column and check
                    # whether this point is over the previous center point or not. If the answer is yes so the
                    # center if found. If no, Ignore this row and check next row.

                    if(rowFIP and (img[pixel_pos_col,col]==0)):
                        rows = pixel_pos_row + np.sum(vector_row_FIP)/2

                        if np.absolute(rows-row)<=8: # allow an error (Number 8 is error and can be modified by the application)
                            number_of_fips = number_of_fips + 1
                            location_FIPS.append([row,col])
                    
                    pixel_pos_row = pixel_pos_row + tr_col[j]
            
            pixel_pos_col = pixel_pos_col + length_modules[i]

    return location_FIPS
    # Since we allocated the first position to 0,0 just remove this

    #location_FIPS =  location_FIPS[2:-1,:]     

def get_Correct_Order_FIPs_Fn(FIPs):
    # A ---------C
    #
    #
    # B

    distancia = pdist(FIPs)
    max_index = np.argmax(distancia)

    # Find top left FIP, "A"
    dic = {
        0:[FIPs[3,:],FIPs[2,:],FIPs[1,:]],
        1:[FIPs[2,:],FIPs[3,:],FIPs[1,:]],
        2:[FIPs[1,:],FIPs[2,:],FIPs[3,:]]
    }

    A,B,C = dic[max_index]

    A=np.array(A)
    B=np.array(B)
    C=np.array(C)

    AB = B-A
    AC = C-A

    k = AB[0]*AC[1] - AB[1]*AC[0]
    #AB _|_ AC > 0, AC left of AB => C is topRight. Otherwise the opposite.

    if k > 0:
        return [B,A,C]
    else:
        return [C,A,B]

def findFIP_order_fn(FIP):
    # This function considers all possible FIP probable locations and return
    # three locations to the "real" FIPs
    df = DataFrame(FIP)
    if(len(FIP) > 2):
        
        points = KMeans(n_clusters=3,init='random',max_iter=5)
        idx = points.labels_
        points.fit(df)
        centroide = points.cluster_centers_
        FIPs = []
        for i in range (0,3,1):
            Pos = centroide[i,:]
            Determnd_location = FIP[idx==i,:]
            distancias = [np.linalg.norm(j-b) for j in a]
            y = distancias.index(min(distancias))
            FIPs.append(Determnd_location[y,:])

        FIPs = get_Correct_Order_FIPs_Fn(FIPs)
    else:
        print("Algo Salio Mal al recuperar el QR")
        
def GetPattern_message_Fn(Im, AP_h_check):

    #make image binary
    img = im_binary(Im)
        #falta filtro medfilt2
    
    #reconocimiento del patron
    FIPs = find_Probable_FIPS_fn(img)
    FIP_L = findFIP_order_fn(FIPs)

    return 0
    #find AP (aligment pattern) puntos en la imagen

# Load an Real QR image
print(cv2.__version__)
image2 = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
cv2.imshow("QR NORTE", image2)
GetPattern_message_Fn(image2,0)  
cv2.waitKey()
cv2.destroyAllWindows()

'''
# Convert RGB -> Grayscale
img_grayscale = cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)
# Binarization
img_binaria = im_binary(img_grayscale)
# Reescalamiento de las imagenes binarizadas
img_resized2 = im_resize(img_binaria,200,cv2.INTER_AREA)
# Filtro de media -> para eliminar ruido salt&papper
img_filtrada = sp.medfilt2d(img_resized2, kernel_size=(3,3))
#cv2.imshow('image gris',img_grayscale)
#cv2.imshow('image binarizada 2 - reescalada',img_resized2)
#cv2.imshow('image filtrada ',img_filtrada)

#img_np = np.ones((1,img_filtrada.shape[0]))
#print(f"longitud array:{img_np.shape} ")
#cv2.imshow('img np', img_np)
cv2.imshow("asda", img_filtrada[1,:])
cv2.waitKey(0)
cv2.destroyAllWindows()
'''




    