import cv2, time
import sys
import numpy as np
import matplotlib.pyplot as plt
import math

GREEN = (0,255,0)
BLUE = (255,0,0)
RED = (0,0,255)
ORANGE = (0,128,255)
YELLOW = (0,255,255)
LIGHT_BLUE = (255,255,0)
PURPLE = (205,0,205)
WHITE = (255,255,255)
BLACK = (0,0,0)

sample_number = 360

offset_index = 0

def extract_feature( image, contours ):
    
    
    height, width, channel = image.shape
     

    c_list_d = []
    c_list = []
    min_contour_len = len(contours[0])
    
    for i in xrange(len(contours)):
        if len(contours[i]) < min_contour_len :
            min_contour_len = len(contours[i])
    
    cor = 0
    for i in xrange(len(contours)):
        tmp_list = []
        #print len(contours[i])
        if(len(contours[i])<10):
            continue
       
        M = cv2.moments(contours[i])
        if M['m00']==0:
            cor += 1
            continue
         
        c_list.append(contours[i])
        
        cx = (M['m10']/M['m00'])
        cy = (M['m01']/M['m00'])
        
        max_dis = 0
       
        for c in contours[i]:
            #print c[0],(cx,cy),Eucl_distance(c[0],(cx,cy))
            
            # the (0,0) in image is the left top point
            v1 = ( c[0][0]-cx, (cy-c[0][1]) )
            v2 = (0,height)
            
            if v1[0] == 0 and v1[1] >= 0 :
                angle = 0.0
            elif v1[0] == 0 and v1[1] < 0 :
                angle = 180.0
            else:
                angle = np.arccos( np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) )
                angle = angle*180/np.pi
                
            if( v1[0] < 0 ):
                angle = 360 - angle       
           
            if Eucl_distance(c[0],(cx,cy)) > max_dis:
                max_dis = Eucl_distance(c[0],(cx,cy))
            tmp_list.append( { 'distance':Eucl_distance(c[0],(cx,cy)),'angle' : angle, 'coordinate' : c[0] } )
        #print tmp_list  
        
        for t in tmp_list:
            t['distance'] = t['distance']/max_dis
            
            
        ellipse = cv2.fitEllipse(contours[i])     
        
        tmp_list = rotate_contour( tmp_list, ellipse[2])
        
        #img_copy = image.copy()
        #draw_rotate_contours = np.array(list(contours[i][offset_index:]) + list(contours[i][:offset_index]))
        #for c in draw_rotate_contours:
            #cv2.drawContours(img_copy, [c], -1, (0,255,0), 3)
            #cv2.imshow('img',img_copy)
            #cv2.waitKey(5)     
            
        distance_list, coordinate_list = sample_by_angle(tmp_list, sample_number)
        #if len(distance_list) != 360 :
            #print 'len(distance_list):',len(distance_list)
        c_list_d.append( distance_list )
        
        #img_copy = image.copy()
        #draw_rotate_contours = np.array(list(contours[i][offset_index:]) + list(contours[i][:offset_index]))
        #for c in coordinate_list:
            #cv2.drawContours(img_copy, [np.array([c])], -1, (0,255,0), 3)
            #cv2.imshow('img',img_copy)
            #cv2.waitKey(5)         
        
        #print sample_by_angle(tmp_list, sample_number) 
    # end contour for
    cnt_rgb_list = []
    cnt_lab_list = []
    cnt_hsv_list = []
    cnt_intensity_list = []
    size_list = []
    max_size = len( max(c_list,key=lambda x:len(x)) )
    for cnt in c_list : 
        #cnt_rgb_list.append( FindCntAvgRGB(cnt, image) )
        #cnt_rgb_list.append( FindCntRgbHis(cnt, image) )
        #cnt_lab_list.append( FindCntLabHis(cnt, image) )
        #cnt_hsv_list.append( FindCntHsvHis(cnt, image) )
        cnt_intensity_list.append( FindCntAvgColorInetnsity(cnt, image) )
        size_list.append( [len(cnt)/float(max_size)] )
        #size_list.append( [len(cnt)] )
    #print 'cor err:',cor    
    return  c_list, c_list_d, cnt_intensity_list, size_list


def rotate_contour( contour_list, main_angle ):
    #print main_angle
  
    min_distance = 1000
    min_index = 0
    angle_offset = 0
    dis_0 = 1000
    dis_180 = 1000
    angle_0_dis = 0
    angle_180_dis = 0
    angle_0 = 0
    angle_180 = 0
    index_0 = 0
    index_180 = 0
    
    for i in xrange( len(contour_list) ):
        
        if  abs( contour_list[i]['angle']-main_angle ) < dis_0 :
            dis_0 = abs(contour_list[i]['angle']-main_angle )
            angle_0_dis = contour_list[i]['distance']
            index_0 = i
            angle_0 = contour_list[i]['angle']
        
        if  abs( contour_list[i]['angle']-main_angle-180 ) < dis_180 :
            dis_180 = abs(contour_list[i]['angle']-main_angle-180 )
            angle_180_dis = contour_list[i]['distance']  
            index_180 = i
            angle_180 = contour_list[i]['angle']
        
        if angle_0_dis < angle_180_dis :
            min_index = index_0
            angle_offset = angle_0
        else:
            min_index = index_180
            angle_offset = angle_180     
        
        offset_index = min_index
        
        #if ( abs(contour_list[i]['angle']-main_angle) < 1 or abs(contour_list[i]['angle']-main_angle-180) < 1 ) and contour_list[i]['distance'] < min_distance :  
            #min_distance = contour_list[i]['distance']
            #min_index = i 
            #angle_offset = contour_list[i]['angle']
            #offset_index = min_index
            
    rotate_list = contour_list[min_index:]+contour_list[:min_index]
    
    for i in xrange( len(rotate_list) ):
        rotate_list[i]['angle'] = rotate_list[i]['angle'] - angle_offset
        if rotate_list[i]['angle'] < 0 :
            rotate_list[i]['angle'] += 360
    
    #n_180 = 0
    #offset = rotate_list[0]['angle']
    #tmp = offset
    #rotate_list[0]['angle']=0
   
    #for i in xrange( 1, len(rotate_list) ):
        #if tmp - rotate_list[i]['angle'] > 100 :
            #n_180+=1
        #tmp = rotate_list[i]['angle']
        #rotate_list[i]['angle'] = rotate_list[i]['angle']-offset+180*n_180
    
    return rotate_list 

def sample_by_angle( contour_list, n_sample ):
    
    angle_hash = []
    sample_list = []
    angle_err = 0.3
    tmp_i = 0

    per_angle = 360.0/n_sample
    
    for angle in np.arange(0,360,per_angle):
        #print angle
        index = -1
        deviation = 10
        sample_angle = 0
        sample_distance = 0
        sample_coordinate = 0
        
        angle_match = False
        for i in xrange( tmp_i, len(contour_list) ):
              
            #if angle == 0:
                #print "contour_list[i]['angle']:",contour_list[i]['angle'],"abs(contour_list[i]['angle']-angle):",abs(contour_list[i]['angle']-angle)
            #tmp_i = i
            #print "contour_list[i]['angle']",contour_list[i]['angle'],"angle",angle,"abs(contour_list[i]['angle']-angle)",abs(contour_list[i]['angle']-angle),"deviation",deviation
            if abs(contour_list[i]['angle']-angle) < angle_err and abs(contour_list[i]['angle']-angle) < deviation :
                angle_match = True
                deviation = abs(contour_list[i]['angle']-angle)
                index = i
                sample_angle = contour_list[i]['angle']
                sample_distance = contour_list[i]['distance']                                
                sample_coordinate = contour_list[i]['coordinate']
                
            #elif index >=0 :
                #sample_list.append( { 'distance':contour_list[i-1]['distance'], 'angle':contour_list[i-1]['angle'] } )
                #angle_hash.append(angle)
                #break
        if angle_match:
            angle_hash.append(angle)
            sample_list.append( { 'distance':sample_distance, 'angle':sample_angle, 'coordinate':sample_coordinate } )
        # end contour_list for 
    # end angle for
    
    distance_list = []
    coordinate_list = []
    angle_hash.append(360.0)
    sample_list.append( { 'distance':sample_list[0]['distance'], 'angle':360.0, 'coordinate':contour_list[0]['coordinate'] } )
    
    # use interpolat to complete the sample angle distance
    for i in xrange( len(angle_hash)-1 ):
        distance_list.append( sample_list[i]['distance'] )
        coordinate_list.append( sample_list[i]['coordinate'] )
        #print 'out:',angle_hash[i]
        for sample_i in np.arange( angle_hash[i]+per_angle, angle_hash[i+1], per_angle ):      
            #print 'in:',sample_i
            #print angle_hash[i],sample_list[i]['distance'],angle_hash[i+1],sample_list[i+1]['distance'],sample_i
            #print Interpolation( angle_hash[i], sample_list[i]['distance'], angle_hash[i+1], sample_list[i+1]['distance'], sample_i )
            distance_list.append( Interpolation( angle_hash[i], sample_list[i]['distance'], angle_hash[i+1], sample_list[i+1]['distance'], sample_i ) )
            coordinate_list.append( Interpolation_coordinate( angle_hash[i], sample_list[i]['coordinate'], angle_hash[i+1], sample_list[i+1]['coordinate'], sample_i ) )
    #if len(distance_list) != 360 :
        #print distance_list
    #print 'len(coordinate_list):',coordinate_list
    return distance_list, coordinate_list

def Eucl_distance(a,b):
    
    if type(a) != np.ndarray :
        a = np.array(a)
    if type(b) != np.ndarray :
        b = np.array(b)
    
    return np.linalg.norm(a-b)  


def Interpolation( a, a_d, b, b_d, i ):
    return ( abs(i-a)*b_d + abs(b-i)*a_d ) / float(abs(b-a)) 

def Interpolation_coordinate( a, a_d, b, b_d, i ):  
    return np.array( [ int(round( a_d[0]+ (b_d[0]-a_d[0])*( float(i-a)/(b-a) ) )) , int(round( a_d[1] + (b_d[1]-a_d[1])*( float(i-a)/(b-a) ) ) ) ] )

def FindCntAvgColorInetnsity( cnt, img ):
    
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[:] = 0
    cnt = cv2.convexHull( np.array(cnt))
    cv2.drawContours( mask, [cnt], -1, 255, -1 )
    cv2.drawContours( mask, [cnt], -1, 0, 1 )    
   
    avg = [0.0,0.0,0.0]
    img_lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    cnt_lab = img_lab[mask==255]
    num = len(cnt_lab)
    
    if num < 1 :
        return [0.0,0.0,0.0]
            
    avg_lab = [ 0.0, 0.0, 0.0 ]
    for lab in cnt_lab : 
        #print rgb
        avg_lab[0] += lab[0]
        avg_lab[1] += lab[1]
        avg_lab[2] += lab[2]
   
    for i in xrange( len(avg_lab) ):
        avg_lab[i] /= float(num)
    
    # count color intensity by A, B (LAB)
    intensity = math.sqrt( pow(avg_lab[1],2) + pow(avg_lab[2],2)  )
    
    #return intensity
    return avg_lab[:]

def FindCntHsvHis( cnt, img ):
    
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[:] = 0
       
    cv2.drawContours( mask, [cnt], -1, 255, -1 )
    cv2.drawContours( mask, [cnt], -1, 0, 1 )    
   
    avg = [0.0,0.0,0.0]
    img_hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV)
    cnt_hsv = img_hsv[mask==255]
    num = len(cnt_hsv)
    
    if num == 0: 
        mask[:] = 0
        cnt = cv2.convexHull( np.array(cnt))
        cv2.drawContours( mask, [cnt], -1, 255, -1 ) 
        cv2.drawContours( mask, [cnt], -1, 0, 1 )
        cnt_hsv = img_hsv[mask==255]
        num = len(cnt_hsv)
        
    H_scale = [0]*256
    S_scale = [0]*256
    V_scale = [0]*256
    
    for hsv in cnt_hsv : 
        #print rgb
        H_scale[hsv[0]] += 1
        S_scale[hsv[1]] += 1
        V_scale[hsv[2]] += 1
    
    maxH = max(H_scale)
    maxS = max(S_scale)
    maxV = max(V_scale)
    
    for i in xrange(256):
        H_scale[i] /= float(maxH)
        S_scale[i] /= float(maxS)
        V_scale[i] /= float(maxV)  
    
    return H_scale + S_scale + V_scale

def FindCntLabHis( cnt, img ):
    
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[:] = 0
       
    cv2.drawContours( mask, [cnt], -1, 255, -1 )
    cv2.drawContours( mask, [cnt], -1, 0, 1 )    
   
    avg = [0.0,0.0,0.0]
    img_lab = cv2.cvtColor( img, cv2.COLOR_BGR2LAB)
    cnt_lab = img_lab[mask==255]
    num = len(cnt_lab)
    
    if num == 0: 
        mask[:] = 0
        cnt = cv2.convexHull( np.array(cnt))
        cv2.drawContours( mask, [cnt], -1, 255, -1 ) 
        cv2.drawContours( mask, [cnt], -1, 0, 1 )
        cnt_lab = img_lab[mask==255]
        num = len(cnt_lab)
        
    L_scale = [0]*256
    A_scale = [0]*256
    B_scale = [0]*256
    
    for lab in cnt_lab : 
        #print rgb
        L_scale[lab[0]] += 1
        A_scale[lab[1]] += 1
        B_scale[lab[2]] += 1
    
    maxL = max(L_scale)
    maxA = max(A_scale)
    maxB = max(B_scale)
    
    for i in xrange(256):
        L_scale[i] /= float(maxL)
        A_scale[i] /= float(maxA)
        B_scale[i] /= float(maxB)  
    
    return A_scale + B_scale

def FindCntRgbHis( cnt, img ):
    
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[:] = 0
       
    cv2.drawContours( mask, [cnt], -1, 255, -1 )
    cv2.drawContours( mask, [cnt], -1, 0, 1 )    
   
    avg = [0.0,0.0,0.0]
    cnt_rgb = img[mask==255]
    num = len(cnt_rgb)
    
    if num == 0: 
        mask[:] = 0
        cnt = cv2.convexHull( np.array(cnt))
        cv2.drawContours( mask, [cnt], -1, 255, -1 ) 
        cv2.drawContours( mask, [cnt], -1, 0, 1 )
        cnt_rgb = img[mask==255]
        num = len(cnt_rgb)
        
    R_scale = [0]*256
    G_scale = [0]*256
    B_scale = [0]*256
    
    for rgb in cnt_rgb : 
        #print rgb
        R_scale[rgb[0]] += 1
        G_scale[rgb[1]] += 1
        B_scale[rgb[2]] += 1
    
    maxR = max(R_scale)
    maxG = max(G_scale)
    maxB = max(B_scale)
    
    for i in xrange(256):
        R_scale[i] /= float(maxR)
        G_scale[i] /= float(maxG)
        B_scale[i] /= float(maxB)  
    
    return R_scale + G_scale + B_scale

def FindCntAvgRGB( cnt, img ):
    
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[:] = 0
    
    
    cv2.drawContours( mask, [cnt], -1, 255, -1 )
    cv2.drawContours( mask, [cnt], -1, 0, 1 )    
   
    avg = [0.0,0.0,0.0]
    cnt_rgb = img[mask==255]
    num = len(cnt_rgb)
    
    if num == 0: 
        mask[:] = 0
        cnt = cv2.convexHull( np.array(cnt))
        cv2.drawContours( mask, [cnt], -1, 255, -1 ) 
        cv2.drawContours( mask, [cnt], -1, 0, 1 )
        cnt_rgb = img[mask==255]
        num = len(cnt_rgb)
        
    for rgb in cnt_rgb : 
        #print rgb
        avg[0] += rgb[0]
        avg[1] += rgb[1]
        avg[2] += rgb[2]
 
    return [ float(avg[0])/(num*255), float(avg[1])/(num*255), float(avg[2])/(num*255) ]
    #return [ float(avg[0])/(num), float(avg[1])/(num), float(avg[2])/(num) ]
       
    
    
#if __name__ == "__main__":
    #start = time.time()
    #main()
    #print 'Total time : ',time.time() - start ,'s'
    #print 'All finished!'