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
import math
# PATH TO IMAGES
dirname = os.path.abspath('./documents/qr_images/real_qr/') 
image= os.path.join(dirname,'qr_type1_north.jpg')

def main():
    
    img_original = cv2.imread(image)

    ret, otsu = cv2.threshold(cv2.cvtColor(img_original,cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    contornos, herencia = cv2.findContours(otsu,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(img_original, contornos, -1,(0,255,0),1)

    print(len(contornos))
    centro_masa = np.empty([len(contornos),2])
    i = 0
    for c in contornos:
        
        momentos = cv2.moments(c)
        if momentos["m00"] == 0:
            centro_masa[i,:] = (0,0)
        else:
            centro_masa[i,:]= (momentos["m10"]/momentos["m00"],
                            momentos["m01"]/momentos["m00"])    

        img_original = cv2.circle(img_original,(int(centro_masa[i,0]),int(centro_masa[i,1])),1,(0,0,255),1)                        
        i = i+1

    i=0
    for c in contornos:
        k=i
        a=0

    cv2.imshow("QR NORTE", img_original)

    cv2.waitKey(0)

    #cv2.destroyAllWindows()

main()
