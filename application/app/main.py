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
    print(im_out)
    k = 0

    for i in range(0,len(img)-1,1):
        if img[i] == img[i+1]:
            im_out[k] = im_out[k] + 1
        else:
            k = k + 1
        print(f"im out{im_out}")
    im_out =  im_out[0:k+1]
    print(f'resultado final {im_out}')
    return im_out

    

def locationFIPs(img):
    number_of_fips = 0
    location_FIPS = [0,0]
    width = int(img.shape[1])
    height = int(img.shape[0])
    dim = (width,height)

    for row in dim:
        line = img[row,:] #pick out the entire row
        
        # calculate the appearence of the modules in row.
        length_modules = 1

a = np.array((0,0,1,0,1,1,1,0,0))
print (a)
res = mod_Tr_Fn(a)
print(res)

'''


# Load an Real QR image
img_color = cv2.imread(image,cv2.IMREAD_COLOR)
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

img_np = np.ones((1,img_filtrada.shape[0]))
print(f"longitud array:{img_np.shape} ")
cv2.imshow('img np', img_np)
cv2.imshow("asda", img_filtrada[1,:])
cv2.waitKey(0)
cv2.destroyAllWindows()

'''



    