import cv2
import numpy as np
import scipy.signal as sp
import matplotlib.pylab as plt
import os

# PATH TO IMAGES
dirname = os.path.abspath('./documents/qr_images/real_qr/') 
image= os.path.join(dirname,'qr_bottle.jpeg')
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
    im_out = np.ones((img.shape[0]),dtype=int)
    print(img)
    k = 0

    for i in range(0,len(img)-1,1):
        if img[i] == img[i+1]:
            im_out[k] = im_out[k] + 1
        else:
            k = k + 1
        #print(f"im out{im_out}")
    im_out =  im_out[0:k+1]
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
    if (np.absolute(np.multiply(module_size,Pat)-A) <= maxV):
        out = True
    else:
        out = False
    
    return out, module_size

def find_Probable_FIPS_fn(img):
    pattern = np.array((1,1,3,1,1))
    number_of_fips = 0
    ################convertir location_FIPS a lista
    #location_FIPS = []
    #location_FIPS.append([row,col])
    location_FIPS = [0,0]
    width = int(img.shape[1])
    height = int(img.shape[0])
    dim = (width,height)

    for row in range(0,width,1):
        tr_line = img[row,:] #pick out the entire row
        
        # calculate the appearence of the modules in row.
        length_modules = mod_Tr_Fn(tr_line)

        pixel_pos_col = 1 # check sequence to find approbriate pattern similar to finder patterns

        for i in range (0,length_modules-4,1):
            vectorFIP = length_modules[i:i+4]
            (isFIP,moduleSize) = checkRatio_Fn(vectorFIP,pattern)

            # The ratio is correct and the first value is black, we should search in that column
            if ((isFIP & img[row,pixel_pos_col]) == 0):
                col = pixel_pos_col + np.floor(np.sum(vectorFIP)/2) # Specific column for search

                pixel_pos_row = 1 # Punto inicial
                
                # Busca a traves de todas las columnas y ve si el patron FIP es encontrado
                # si lo es, lo almacena como candidato
                tr_col = mod_Tr_Fn(img[:,row])

                for j in range(0,len(tr_col)-4,1):
                    vector_row_FIP = tr_col[j:j+4]
                    rowFIP, _ = checkRatio_Fn(vector_row_FIP, pattern)

                    if((rowFIP&img[pixel_pos_col,col])==0):
                        rows = pixel_pos_row + np.sum(vector_row_FIP)/2

                        if np.absolute(rows-row)<=8: # allow an error (Number 8 is error and can be modified by the application)
                            number_of_fips = number_of_fips + 1
                            location_FIPS  = [[location_FIPS],[row,col]]
                    
                    pixel_pos_row = 


                

a = np.array((0,0,1,0,1,1,1,0,0))
b = np.array((1,1,3,1,1))
c = 2
print(np.multiply(b,c))

#res = mod_Tr_Fn(a)
#print(res)

# Load an Real QR image
img_color = cv2.imread(image,cv2.IMREAD_COLOR)
cv2.imshow("asdasdsa", img_color[:,5:50])
cv2.waitKey(0)
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




    