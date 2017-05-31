import cv2, os, math
import numpy as np

def find_gradient_img(img):

    resize_para = 736.0
    
    kernel = np.array( 1*[[-1,-1,-1],
                          [-1, 8,-1],
                          [-1,-1,-1]]  )
    kernel1 = np.array( [[-1, 0, 0],
                        [ 0, 0, 0],
                        [ 0, 0, 1]]  )
    kernel2 = np.array( [[0,-1, 0],
                        [0, 0, 0],
                        [0, 1, 0]]  )
    kernel3 = np.array( [[0, 0,-1],
                        [0, 0, 0],
                        [1, 0, 0]]  )
    kernel4 = np.array( [[0, 0,0],
                        [-1, 0, 1],
                        [0, 0, 0]]  )
    
    
    #img = cv2.imread('brick (23).jpg')
    
    height, width = img.shape[:2]
    img = cv2.resize( img, (0,0), fx = resize_para/height, fy = resize_para/height )
    
    lab = cv2.cvtColor( img, cv2.COLOR_BGR2LAB)
    lab_l = lab[:,:,0]
    lab_a = lab[:,:,1]
    lab_b = lab[:,:,2]
    
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #img = gray
    
    lab_list = [ lab_l, lab_a, lab_b ]
    
    gradient_list = []
    for lab_channel in lab_list :
        gradient1 = cv2.filter2D(lab_channel, -1, kernel1)
        gradient2 = cv2.filter2D(lab_channel, -1, kernel2)
        gradient3 = cv2.filter2D(lab_channel, -1, kernel3)
        gradient4 = cv2.filter2D(lab_channel, -1, kernel4)
    
        gradient = gradient1.copy()
        gradient[:] = (abs(gradient1[:]) + abs(gradient2[:]) + abs(gradient3[:]) + abs(gradient4[:]) ) 
        gradient_list.append(gradient)
    
    gradient = lab_b.copy() 
    
    h,w = gradient.shape[:2]
    for x in range(h):
        for y in range(w):
            gradient[x,y] = math.sqrt( pow( gradient_list[0][x,y],2 ) +  pow( gradient_list[1][x,y],2 ) + pow( gradient_list[2][x,y],2 ) )
    
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #gradient = clahe.apply(gradient) 
    
    #gradient = cv2.equalizeHist(gradient)
    
    #combine_image = np.concatenate((img, img_copy), axis=1) 
    
    #cv2.imshow( 'gradient', gradient )
    #cv2.waitKey(0)
    return gradient

