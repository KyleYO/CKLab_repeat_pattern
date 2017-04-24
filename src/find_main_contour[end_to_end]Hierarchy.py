import numpy as np
import cv2
import os 
import time
import get_contour_feature
import dbscan_cluster
import operator
from operator import itemgetter
import cluster_evaluate
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#import dbclasd

resize_height = 736.0
split_n_row = 16
split_n_column = 16

_local = True
_equalization = False
_sharpen = True
_check_overlap = False

#input_path = '../../first_state_data/'
input_path = '../../input_image/'
#input_path = './input_3para-1/'
#input_path = './input/'
output_path = './output_hierarchy/'
#output_path = './output-Order-varyStd/'

GREEN = (0,255,0)
BLUE = (255,0,0)
RED = (0,0,255)
ORANGE = (0,128,255)
YELLOW = (0,255,255)
LIGHT_BLUE = (255,255,0)
PURPLE = (205,0,205)
WHITE = (255,255,255)
BLACK = (0,0,0)
switchColor = [(255,255,0),(255,0,255),(0,255,255),(255,0,0),(0,255,0),(255,128,0),(255,0,128),(128,0,255),(128,255,0),(0,128,255),(0,255,128)]

resize_ratio = 0.6

def main():
    
    if _local : 
        g_l = '_local'
    else:
        g_l = '_global'
    
    file_n = 0
    switch_i = 0

    max_time_img = ''
    min_time_img = ''
    min_time = 99999.0
    max_time = 0.0
    
    for i,fileName in enumerate(os.listdir(input_path)):
        
        if(fileName[-3:]!='jpg' and fileName[-3:]!='JPG' and fileName[-3:]!='jpeg' and fileName[-3:]!='png'):
            print "Wrong format file: "+fileName
            continue   
        
        start_time = time.time()
        file_n = i
        each_scale_best_result = []        
        
        fileName = 'test1.png'  
        
        print 'Input:',fileName
        
        image = cv2.imread( input_path + fileName )
      
        height, width, channel = image.shape
        image_ori = cv2.resize( image, (0,0), fx= resize_height/height, fy= resize_height/height)
        
        hsv = cv2.cvtColor( image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor( image, cv2.COLOR_BGR2LAB)
        
        h = hsv[:,:,0]
        s = hsv[:,:,1]
        v = hsv[:,:,2]
        
        l = lab[:,:,0]
        a = lab[:,:,1]
        b = lab[:,:,2]
        
        _r = image[:,:,0]
        _g = image[:,:,1]
        _b = image[:,:,2]
        
       
  
        
       
        ratio = 0.4
        ratio = 280/ float( image_ori.shape[0] )
        
        combine_imagergb = np.concatenate((_r, _g), axis=1)  
        combine_imagergb = np.concatenate((combine_imagergb, _b), axis=1)
            
        
        combine_image = np.concatenate((image,  hsv), axis=1)  
        combine_imagehsv = np.concatenate((h, s), axis=1)  
        combine_imagehsv = np.concatenate((combine_imagehsv, v), axis=1)
        #cv2.imshow('img_hsv',cv2.resize(combine_image,(0,0),fx = ratio ,fy=ratio ))
        #cv2.imshow('img_hsv_split',cv2.resize(combine_imagehsv,(0,0),fx = ratio ,fy=ratio ))
        #cv2.waitKey(0)
        
        combine_image = np.concatenate((image,  lab), axis=1)  
        combine_imagelab = np.concatenate((l, a), axis=1)  
        combine_imagelab = np.concatenate((combine_imagelab, b), axis=1)
        #cv2.imshow('img_lab',cv2.resize(combine_image,(0,0),fx = ratio ,fy=ratio ))
        #cv2.imshow('img_lab_split',cv2.resize(combine_imagelab,(0,0),fx = ratio ,fy=ratio ))
        #cv2.waitKey(0)  
        
        combine_image_all = np.concatenate((combine_imagergb, combine_imagehsv), axis=0) 
        combine_image_all = np.concatenate((combine_image_all, combine_imagelab), axis=0)
        #cv2.imshow('img_split',cv2.resize(combine_image_all,(0,0),fx = ratio ,fy=ratio ))
        #cv2.waitKey(0)  
        #cv2.imwrite( output_path + fileName[:-4] +'-rgb-hsv-lab.jpg', combine_image_all)
        #if True:
            #continue
        #image_ori = image.copy()
        
        for j in xrange(2):
            
            scale = 1.0/(2**j)
            print 'Scale:', scale           
            str_scale = '1_'+str(2**j)
            image = cv2.resize( image_ori, (0,0), fx= scale, fy= scale)        
            image = cv2.GaussianBlur(image, (5, 5),0)
               
            if _sharpen :
                print 'Sharpening'
                image = Sharpen(image)
                  
            re_height, re_width = image.shape[:2]
                      
            offset_r = re_height/split_n_row
            offset_c = re_width/split_n_column
           
            if _local :
                
                print 'Detect edge'
                edged = np.zeros(image.shape[:2], np.uint8) 
                edged_hsv = np.zeros(image.shape[:2], np.uint8) 
                edged_rgb = np.zeros(image.shape[:2], np.uint8)                 
                edged_r = edged_rgb.copy()
                edged_g = edged_rgb.copy()
                edged_b = edged_rgb.copy()
                #edged_lab = np.zeros(image.shape[:2], np.uint8) 
                binary = np.zeros(image.shape[:2], np.uint8) 
                
                for row_n in np.arange(0,split_n_row,0.5):
                    for column_n in np.arange(0,split_n_column,0.5):
                        
                        r_l =  int(row_n*offset_r)
                        r_r = int((row_n+1)*offset_r)
                        c_l = int(column_n*offset_c)
                        c_r = int((column_n+1)*offset_c)
                        if row_n == split_n_row-0.5 :
                            r_r = int(re_height)
                        if column_n == split_n_column-0.5 :
                            c_r = int(re_width)    
                            
                        gray = cv2.cvtColor( image[ r_l : r_r , c_l : c_r ], cv2.COLOR_BGR2GRAY) 
                        hsv = cv2.cvtColor( image[ r_l : r_r , c_l : c_r ], cv2.COLOR_BGR2HSV)
                        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
                        h = hsv[:,:,0]
                        s = hsv[:,:,1]
                        v = hsv[:,:,2]   
                        h = cv2.GaussianBlur(h, (5, 5), 0)
                        s = cv2.GaussianBlur(s, (5, 5), 0)
                        v = cv2.GaussianBlur(v, (5, 5), 0)
                        high_thresh_h, bi_h = cv2.threshold(h,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        high_thresh_s, bi_s = cv2.threshold(s,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        high_thresh_v, bi_v = cv2.threshold(v,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        _b = image[ r_l : r_r , c_l : c_r ][:,:,0]
                        _g = image[ r_l : r_r , c_l : c_r ][:,:,1]
                        _r = image[ r_l : r_r , c_l : c_r ][:,:,2]   
                        _b = cv2.GaussianBlur(_b, (5, 5), 0)
                        _g = cv2.GaussianBlur(_g, (5, 5), 0)
                        _r = cv2.GaussianBlur(_r, (5, 5), 0)
                        high_thresh_b, bi_b = cv2.threshold(_b,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        high_thresh_g, bi_g = cv2.threshold(_g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        high_thresh_r, bi_r = cv2.threshold(_r,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)                        
                        
                        #lab = cv2.cvtColor( image[ r_l : r_r , c_l : c_r ], cv2.COLOR_BGR2LAB)
                        #lab = cv2.GaussianBlur(lab, (5,5), 0)
                        #l = lab[:,:,0]
                        #a = lab[:,:,1]
                        #b = lab[:,:,2]
                        #l = cv2.GaussianBlur(l, (5, 5), 0)
                        #a = cv2.GaussianBlur(a, (5, 5), 0)
                        #b = cv2.GaussianBlur(b, (5, 5), 0)  
                        #high_thresh_l, bi_l = cv2.threshold(l,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        #high_thresh_a, bi_a = cv2.threshold(a,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        #high_thresh_b, bi_b = cv2.threshold(b,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)                        
                        
                        
                        #LAB = cv2.cvtColor( image[ r_l : r_r , c_l : c_r ], cv2.COLOR_BGR2LAB)
                        #gray = LAB2Gray(LAB)
                        
                        #cv2.imshow('gray',gray)
                        #cv2.waitKey(0)                         
                        
                      
                        gray = cv2.GaussianBlur(gray, (5, 5), 0)
                        if _equalization:
                            gray = cv2.equalizeHist(gray)
                        #print gray
                        #cv2.imshow('gray',gray)
                        #cv2.waitKey(0)                        
                        
                        high_thresh,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        low_thresh = 0.5 * high_thresh  
                        #ROI all background
                        if high_thresh < 20 :
                            continue
                        #if high_thresh_s < 20 :
                            #continue                        
                        #print 'high_thresh:',high_thresh
                        
                        
                        edged[ r_l : r_r , c_l : c_r ] = edged[ r_l : r_r , c_l : c_r ] | cv2.Canny(gray, low_thresh, high_thresh)
                        #edged_hsv[ r_l : r_r , c_l : c_r ] = edged_hsv[ r_l : r_r , c_l : c_r ] | cv2.Canny(h, high_thresh_h*0.5, high_thresh_h) | cv2.Canny(s, high_thresh_s*0.5, high_thresh_s) | cv2.Canny(v, high_thresh_v*0.5, high_thresh_v)
                        #edged_hsv[ r_l : r_r , c_l : c_r ] = edged_hsv[ r_l : r_r , c_l : c_r ] | cv2.Canny(s, high_thresh_s*0.5, high_thresh_s)| cv2.Canny(v, high_thresh_v*0.5, high_thresh_v) 
                        #edged_hsv[ r_l : r_r , c_l : c_r ] = edged_hsv[ r_l : r_r , c_l : c_r ] | cv2.Canny(h, high_thresh_h*0.5, high_thresh_h) 
                        #edged_hsv[ r_l : r_r , c_l : c_r ] = edged_hsv[ r_l : r_r , c_l : c_r ] | cv2.Canny(s, high_thresh_s*0.5, high_thresh_s) 
                        edged_hsv[ r_l : r_r , c_l : c_r ] = edged_hsv[ r_l : r_r , c_l : c_r ] | cv2.Canny(v, high_thresh_v*0.5, high_thresh_v) 
                        #edged_lab[ r_l : r_r , c_l : c_r ] = edged_lab[ r_l : r_r , c_l : c_r ] | cv2.Canny(a, high_thresh_a*0.5, high_thresh_a)| cv2.Canny(b, high_thresh_b*0.5, high_thresh_b)
                       
                        if high_thresh_r > 20 :
                            edged_r[ r_l : r_r , c_l : c_r ] = edged_r[ r_l : r_r , c_l : c_r ] | cv2.Canny(_r, high_thresh_r*0.5, high_thresh_r) 
                        if high_thresh_g > 20 :
                            edged_g[ r_l : r_r , c_l : c_r ] = edged_g[ r_l : r_r , c_l : c_r ] | cv2.Canny(_g, high_thresh_g*0.5, high_thresh_g)
                        if high_thresh_b > 20 :
                            edged_b[ r_l : r_r , c_l : c_r ] = edged_b[ r_l : r_r , c_l : c_r ] | cv2.Canny(_b, high_thresh_b*0.5, high_thresh_b)
                        
                        edged_rgb[ r_l : r_r , c_l : c_r ] = edged_rgb[ r_l : r_r , c_l : c_r ] | edged_r[ r_l : r_r , c_l : c_r ] | edged_g[ r_l : r_r , c_l : c_r ] | edged_b[ r_l : r_r , c_l : c_r ]
                        
                        #binary[ r_l : r_r , c_l : c_r ] = th2                     
                        #cv2.imshow('gray',gray)
                        #cv2.imshow('edged',edged)
                        #cv2.waitKey(0)                        
                          
            else:
                gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                if _equalization:
                    gray = cv2.equalizeHist(gray)
                
                #cv2.imshow('gray',gray)
                #cv2.imshow('equalize',gray_g)
                #cv2.waitKey(0)
                high_thresh,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                low_thresh = 0.5 * high_thresh
                edged = cv2.Canny(gray, low_thresh, high_thresh)
                print 'low_thresh:',low_thresh, ' high_thresh:', high_thresh
                
            edged = cv2.resize( edged, (0,0), fx= 1.0/scale, fy= 1.0/scale)
            edged_hsv = cv2.resize( edged_hsv, (0,0), fx= 1.0/scale, fy= 1.0/scale)
            edged_rgb = cv2.resize( edged_rgb, (0,0), fx= 1.0/scale, fy= 1.0/scale)
            edged_r = cv2.resize( edged_r, (0,0), fx= 1.0/scale, fy= 1.0/scale)
            edged_g = cv2.resize( edged_g, (0,0), fx= 1.0/scale, fy= 1.0/scale)
            edged_b = cv2.resize( edged_b, (0,0), fx= 1.0/scale, fy= 1.0/scale)
            #cv2.imshow('binary',cv2.resize( binary, (0,0), fx= resize_ratio, fy= resize_ratio))
            #cv2.imshow('edged_lab',cv2.resize( edged_lab, (0,0), fx= resize_ratio, fy= resize_ratio))
            cv2.imshow('edged_hsv',cv2.resize( edged_hsv, (0,0), fx= resize_ratio, fy= resize_ratio))
            #cv2.imshow('edged_r',cv2.resize( edged_r, (0,0), fx= resize_ratio, fy= resize_ratio))
            #cv2.imshow('edged_g',cv2.resize( edged_g, (0,0), fx= resize_ratio, fy= resize_ratio))
            #cv2.imshow('edged_b',cv2.resize( edged_b, (0,0), fx= resize_ratio, fy= resize_ratio))
            #cv2.imshow('edged_rgb',cv2.resize( edged_rgb, (0,0), fx= resize_ratio, fy= resize_ratio))
            #cv2.imshow('edged',cv2.resize( edged, (0,0), fx= resize_ratio, fy= resize_ratio))
            cv2.imshow('image_ori',cv2.resize( image_ori, (0,0), fx= resize_ratio, fy= resize_ratio))
            cv2.waitKey(0)                    
            
            print 'Find countour'
            contours = cv2.findContours(edged_hsv,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)[-2]
            contours_hsv = cv2.findContours(edged_hsv,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)[-2]
            contours_rgb = cv2.findContours(edged_rgb,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)[-2]
            #contours_lab = cv2.findContours(edged_lab,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)[-2]
            #print contours
            #contour_image_lab = np.zeros(image_ori.shape,np.uint8)
            contour_image_rgb = np.zeros(image_ori.shape,np.uint8)
            contour_image_hsv = np.zeros(image_ori.shape,np.uint8)
            contour_image = np.zeros(image_ori.shape, np.uint8)
            contour_list = []
            
            #cv2.drawContours( contour_image_lab, contours_lab, -1, GREEN, 1 )            
            cv2.drawContours( contour_image_hsv, contours_hsv, -1, GREEN, 1 )
            cv2.drawContours( contour_image, contours, -1, GREEN, 1 )
            #cv2.imshow('contour_image_lab',contour_image_lab)
            #cv2.imshow('contour_image_hsv',contour_image_hsv)
            #cv2.imshow('contour_image',contour_image)
            #cv2.waitKey(0) 
            ind = 0
            contour_image_rgb[:] = BLACK
            #for c in contours_lab :
                #ind = (ind+1) % len(switchColor)
                #if len(c) < 50:
                    #continue
                
                #cv2.drawContours( contour_image_lab, [c], -1, switchColor[ind], 1 )            
            for c in contours:
                ind = (ind+1) % len(switchColor)
                             
                cv2.drawContours( contour_image_rgb, [c], -1, switchColor[ind], 1 )
            ##cv2.imshow('contour_image_ab',contour_image_lab)
            cv2.imshow('contour_image_rgb',contour_image_rgb)
            cv2.waitKey(0)  
            #cv2.imwrite(output_path + fileName[:-4] +'_scale['+str_scale+'].jpg', contour_image_hsv )
            #if True :
                #continue
            noise = 0
                  
            for c in contours:
                
                convexhull_area = cv2.contourArea(cv2.convexHull( np.array(c)) ) 
              
                if convexhull_area == 0 or ( len(c) > 30 and float(len(c)) / convexhull_area > 0.5 ) or ( len(c) <= 30 and float(len(c)) / convexhull_area > 0.75 ): 
                    noise+=1
                    continue
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) > 20 : 
                    continue
                #if len(c) < 100 : 
                    #continue
                    
                contour_list.append(c)
                #cv2.drawContours( contour_image, [c], -1, GREEN, 1 )
            #cv2.imshow('contour_image',contour_image)
            #cv2.imshow('image_ori',image_ori)
            #cv2.waitKey(0)                        
            if _check_overlap :     
                print 'Remove overlap'
                contour_list = CheckOverlap(contour_list)
 
            #print 'contour_list:',len(contour_list),'noise:',noise
            if len(contour_list) == 0 :
                continue
            
            print 'Extract contour feature'
            # Get contour feature
            c_list, cnt_shape_list, cnt_color_list, cnt_size_list = get_contour_feature.extract_feature( image_ori, contour_list )
            
            #cnt_img = image_ori.copy()
            #for i in xrange( len(cnt_shape_list)-1 ):
                #for j in xrange( i+1,len(cnt_shape_list)):
                    #print '--------------------------------'
                    #print 'shape Distance:',cluster_evaluate.Eucl_distance(cnt_shape_list[i], cnt_shape_list[j])
                    #print 'size Distance:',cluster_evaluate.Eucl_distance(cnt_size_list[i], cnt_size_list[j])
                    #print 'color Distance:',cluster_evaluate.Eucl_distance(cnt_color_list[i], cnt_color_list[j])
                    #cv2.drawContours( cnt_img, c_list[i], -1, GREEN, 2 )
                    #cv2.drawContours( cnt_img, c_list[j], -1, RED, 2 )
                    #combine_image = np.concatenate((image_ori, cnt_img), axis=1)  
                    
                    #cv2.imshow('compare',cv2.resize(combine_image,(0,0),fx=1000.0/combine_image.shape[1],fy=1000.0/combine_image.shape[1]))
                    #cv2.waitKey(0)
                    #cnt_img = image_ori.copy()
                    
            # make feature dictionary
            feature_group_list = []
            for i in xrange( len(c_list) ):
                feature_group_list.append( { 'cnt':c_list[i], 'shape':cnt_shape_list[i], 'color':cnt_color_list[i], 'size':cnt_size_list[i] } )
            
            # make it a group
            feature_group_list = [feature_group_list]
            
            contour_image = np.zeros(image_ori.shape, np.uint8)
                         
            para = [ 'size', 'shape' , 'color' ] 
            
            std_dic = { 'size':cluster_evaluate.Standard_deviation( cnt_size_list ), 'shape':cluster_evaluate.Standard_deviation( cnt_shape_list ) , 'color':cluster_evaluate.Standard_deviation( cnt_color_list ) }
           
               
                
            #para = ['shape_list', 'size_list']
            
            
            label_set_list = []
            
            print 'Respectively use shape, color, and size as feature set to cluster'
            # Respectively use shape, color, and size as feature set to cluster
            for para_index in xrange( len(para) ):
                print '--------------------------------'
                print 'para:',para[para_index]
                
                std = std_dic[para[para_index]]
                
                tmp_feature_group_list = []
               
                for feature_group in feature_group_list :
                    tmp_result = []
                   
                    contour_feature_list = []
                    tmp_cnt_list = []
                    for feature_dic in feature_group:         
                        contour_feature_list.append( feature_dic[ para[para_index] ] )
                        tmp_cnt_list.append( feature_dic['cnt'] )
                  
                    contour_image = np.zeros(image_ori.shape, np.uint8)
                    contour_image[:] = BLACK  
                    cv2.drawContours(contour_image,np.array(tmp_cnt_list),-1,RED,2)
                    cv2.imshow(para[para_index], cv2.resize(contour_image,(0,0),fx=720.0/image_ori.shape[0],fy=720.0/image_ori.shape[0]))
                    cv2.waitKey(0)                  
                    # hierarchical clustering
                    label_list = Hierarchical_clustering(contour_feature_list)
                  
                    
                    
                    unique_label, label_counts = np.unique(label_list, return_counts=True)
                    #print "unique_label:",unique_label,'label_counts:',label_counts
                    #label_dic = dict(zip(unique_label, label_counts))
                    #print 'label_dic:',label_dic
                    
                    #max_label = max( label_dic.iteritems(), key=operator.itemgetter(1) )[0]
                    #count = label_dic[max_label] 
                    
                    contour_image = np.zeros(image_ori.shape, np.uint8)
                    contour_image[:] = BLACK  
                                    
                    # split the sub-group
                    splited_feature_group = []
                    group_list = []
                    max_group = []
                    color_index=0
                    for label in unique_label :
                        COLOR = switchColor[ color_index % len(switchColor) ]
                        color_index+=1
                        tmp_group = []
                        tmp_splited_group = []
                        for i in xrange( len(label_list) ):
                            if label_list[i] == label :  
                                cv2.drawContours(contour_image,[feature_group[i]['cnt']],-1,COLOR,2)                                                     
                                tmp_group.append( contour_feature_list[i] )
                                tmp_splited_group.append( feature_group[i] )
                                #if label == max_label:
                                    #max_group.append(contour_feature_list[i])
                        if len(tmp_group) > 0 :
                            group_list.append(tmp_group)
                            splited_feature_group.append( tmp_splited_group )
                        #cv2.imshow(para[para_index]+' splited', cv2.resize(contour_image,(0,0),fx=720.0/image_ori.shape[0],fy=720.0/image_ori.shape[0]))
                        #cv2.waitKey(0)                        
                    tmp_feature_group_list += splited_feature_group
                    cv2.imshow(para[para_index]+' splited', cv2.resize(contour_image,(0,0),fx=720.0/image_ori.shape[0],fy=720.0/image_ori.shape[0]))
                    cv2.waitKey(0)                    
                                        
                # end feature_group for
             
                feature_group_list = tmp_feature_group_list
                #print len(feature_group_list)
                
                contour_image = np.zeros(image_ori.shape, np.uint8)
                contour_image[:] = BLACK               
                image = image_ori.copy()
                color_index = 0
         
                for i in xrange( len(feature_group_list) ) :
                    COLOR = switchColor[ color_index % len(switchColor) ]
                    #contour_image[:] = BLACK
                    for feature_dic in feature_group_list[i] :                   
                        cv2.drawContours( contour_image, [feature_dic['cnt']], -1, COLOR, 2 )
                        cv2.drawContours( image, [feature_dic['cnt']], -1, COLOR, 2 )
                    #cv2.imshow('! : '+ para[para_index], cv2.resize( contour_image, (0,0), fx= resize_ratio, fy= resize_ratio) )
                    #cv2.waitKey(0)                    
                    color_index += 1
                combine_image = np.concatenate((image, contour_image), axis=1)  

                cv2.imshow('cluster by : '+ para[para_index], cv2.resize( combine_image, (0,0), fx= resize_ratio, fy= resize_ratio) )
                cv2.waitKey(0)
            
            # end para_index for
            # get the max group and draw
            feature_group_list.sort( key = lambda x: len(x) , reverse = False) 
            #max_feature_group = feature_group_list[-1]
            #other_feature_group = feature_group_list[:-1]

            count = len(feature_group_list[-1])
            
            contour_image = np.zeros(image_ori.shape, np.uint8)
            contour_image[:] = BLACK               
            image = image_ori.copy()
            
            color_index = 0
            for i in xrange( len(feature_group_list) ) :
                COLOR = WHITE
                if i == len(feature_group_list)-1 : # max group
                    COLOR = RED
                else:
                    COLOR = GREEN
                COLOR = switchColor[ color_index % len(switchColor) ]  
                color_index += 1
                for feature_dic in feature_group_list[i] : 
                    cv2.drawContours( contour_image, [feature_dic['cnt']], -1, COLOR, 2 )
                    if i == len(feature_group_list)-1 :
                        cv2.drawContours( image, [feature_dic['cnt']], -1, COLOR, 1 )
           
            combine_image = np.concatenate((image, contour_image), axis=1) 
            
            cv2.imshow(fileName+' count:'+str(count), cv2.resize( combine_image, (0,0), fx = resize_ratio, fy = resize_ratio) ) 
            cv2.waitKey(0)
            
            each_scale_best_result.append( { 'filename':fileName, 'img':combine_image,'count':count } )
         
        #end scale for
        print 'Finished in ',time.time()-start_time,' s'
        for img_tuple in each_scale_best_result:
            print '--------------------------------------------------'
            print img_tuple['filename']
            print 'Total Count : ',img_tuple['count']
            #cv2.imwrite( output_path + img_tuple['filename'] +'_Clu['+str(img_tuple['evaluate_score'])+']_Count['+str(img_tuple['count'])+']'+'.jpg', img_tuple['img'] ) 
            cv2.imshow(img_tuple['filename'], cv2.resize( img_tuple['img'], (0,0), fx= resize_ratio, fy= resize_ratio))
        cv2.waitKey(0)
        
        print '-----------------------------------------------------------'    
        each_img_time = time.time() - start_time
        if each_img_time > max_time : 
            max_time = each_img_time
            max_time_img = fileName
        if each_img_time < min_time :
            min_time = each_img_time
            min_time_img = fileName
            
    print 'img:', max_time_img ,' max_time:',max_time,'s'
    print 'img:', min_time_img ,'min_time:',min_time,'s'
        
    
def Sharpen(img):
    
    kernel_sharpen = np.array([[-1,-1,-1,-1,-1],
                                 [-1,2,2,2,-1],
                                 [-1,2,8,2,-1],
                                 [-1,2,2,2,-1],
                                 [-1,-1,-1,-1,-1]]) / 8.0  
   
    return cv2.filter2D(img, -1, kernel_sharpen) 

def draw_image( image_ori, c_list, label_list, max_label ):
    
    if type(label_list) != np.ndarray :
        label_list = np.array(label_list)
    
    samples_mask = np.zeros_like(c_list, dtype=bool)
    samples_mask[:] = True     
    index_mask = ( label_list == max_label )
    image = image_ori.copy()
    contour_image = np.zeros(image_ori.shape, np.uint8)
    contour_image[:] = BLACK            

    c_list = np.array(c_list)
    for c in c_list[samples_mask ^ index_mask]  : 
        if len(c_list) > 1 :
            cv2.drawContours( contour_image, [c], -1, GREEN, 1 )
        else:
            cv2.drawContours( contour_image, c_list, -1, GREEN, 1 )

    tmp_c = []
    drawed_list = []
    max_contour_list = list(c_list[samples_mask & index_mask])
    max_contour_list.sort( key = lambda x: len(x) , reverse = False) 
    
    #print 'len(max_contour_list):',len(max_contour_list)
    count_loss = 0
    for c in max_contour_list  :  
        if len(max_contour_list) > 1 :
            #if IsOverlapAll( c, drawed_list ) :
                #count_loss += 1
                #tmp_c = c
                #continue               
            cv2.drawContours( contour_image, [c], -1, RED, 1 )                      
            cv2.drawContours( image, [c], -1, RED, 1 )
            drawed_list.append(c)
            tmp_c = c
        else:
            cv2.drawContours( contour_image, max_contour_list, -1, RED, 1 )
            cv2.drawContours( image, max_contour_list, -1, RED, 1 )                    
        
    combine_image = np.concatenate((image, contour_image), axis=1)      
    return combine_image, count_loss

def CheckOverlap( cnt_list ):
    
    if cnt_list == []:
        return []
    
    checked_list = []
    # sort list from little to large
    cnt_list.sort( key = lambda x: len(x) , reverse = False)
    
    for c in cnt_list  :  
        if IsOverlapAll( c, checked_list ) : 
            continue               
        checked_list.append(c)    
    
    return checked_list
        
def IsOverlap( cnt1, cnt2 ):
    
    if cnt1 == [] or cnt2 == [] :
        return False
    
    c1M = GetMoment(cnt1)
    c2M = GetMoment(cnt2)
    c1_min_d = MinDistance(cnt1)
    c2_min_d = MinDistance(cnt2)
    moment_d = cluster_evaluate.Eucl_distance( c1M, c2M )
    
    return ( moment_d < c1_min_d or moment_d < c2_min_d ) and max(c1_min_d,c2_min_d)/min(c1_min_d,c2_min_d) <= 3

def IsOverlapAll( cnt, cnt_list ):
    
    if cnt == [] or len(cnt_list) < 1 :
        return False

    for c in cnt_list :
        if len(c) == len(cnt) and GetMoment(c) == GetMoment(cnt):
            #print 'same one'
            continue
        if IsOverlap( cnt, c ) :
            return True
    
    return False

def MinDistance(cnt):
    
    cM = GetMoment(cnt)
    min_d = cluster_evaluate.Eucl_distance( (cnt[0][0][0],cnt[0][0][1]), cM )
    for c in cnt :
        d = cluster_evaluate.Eucl_distance( (c[0][0],c[0][1]), cM ) 
        if d < min_d :
            min_d = d
            
    return min_d

def LAB2Gray(img):
    
    _w, _h, _c = img.shape
    
    gray = np.zeros(img.shape[:2], np.uint8) 

    for i in xrange(_w):
        for k in xrange(_h):
            a = int(img[i][k][1])
            b = int(img[i][k][2])
            gray[i,k] = ( a + b )/2
            
            
    return gray
    
def GetMoment(cnt):
    
    num = len(cnt)
    if num < 2 :
        return cnt
    cx = 0
    cy = 0
    for c in cnt :
        cx += c[0][0]
        cy += c[0][1]
        
    return float(cx)/num, float(cy)/num

def Hierarchical_clustering( cnt_list, cut_method = 'elbow' ):
    
    # hierarchically link cnt by order of distance from distance method 'ward'
    cnt_hierarchy = linkage( cnt_list, 'ward')
    rev = 10
    
    max_d = 10
    if cut_method == 'elbow' or True:
        last = cnt_hierarchy[:, 2]
        print 'last:',last
        last = [ x for x in last if x > 0 ]
        print 'last:',last
        acceleration = np.diff(last) 
        
        #acceleration_rev = acceleration[::-1]
        print 'acceleration:',acceleration 
        
        if len(acceleration) < 2 :
            return [0]*len(cnt_list)
        avg_diff = sum(acceleration)/float(len(acceleration))
        tmp = acceleration[0]
        
        ONE_group = True
        ONE_group = False
        
        cut_point_list = []
        for i in xrange( 1,len(acceleration) ):
            #if acceleration[i] > avg_diff and acceleration[i] > 20 * (tmp/ float(i)) : 
                #max_d = last[i]
                #ONE_group = False
                #print 'cut index:',i
                #break
            if acceleration[i] > avg_diff:
                cut_point_list.append( [ i, acceleration[i]/(tmp/float(i) ) ] )
                
            tmp += acceleration[i]
            
        cut_point_list.sort( key = lambda x : x[1], reverse = True )
        
        print 'cut index:',cut_point_list[0][0]
        max_d = last[cut_point_list[0][0]]
        #max_d = last[acceleration.argmax()]
    #elif cut_method == 'inconsistency':
    
    plt.bar(left=range(len(acceleration)),height=acceleration)   
    plt.show()
    if ONE_group:
        print 'all in one group!'
        return [0]*len(cnt_list) 
    
    print 'acceleration.argmax():',acceleration.argmax()
    clusters = fcluster(cnt_hierarchy, max_d, criterion='distance')   
    print '----------------------------------'
    return clusters
    

if __name__ == '__main__' :
    
    t_start_time = time.time()
   
    main()
    #_local = False
    ##output_path = './output_global/'    
    #main()
    print 'All finished in ',time.time()-t_start_time,' s'
