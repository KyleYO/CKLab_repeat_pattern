import numpy as np
import cv2
import os 
import time
import csv
import math
import get_contour_feature
import operator
from operator import itemgetter
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt



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


resize_height = 736.0
split_n_row = 16
split_n_column = 16
gaussian_para = 3

_sharpen = True
_check_overlap = True
_remove_small_and_big = True
_remove_high_density = True
_remove_too_many_edge = True
_checkConvex = False
_gaussian_filter = True
_use_structure_edge = True
_enhance_edge = True
_gray_value_redistribution_local = True
_record_by_csv = False
_use_comebine_weight = False


input_path = '../../input_image/All_data/'
#input_path = '../../input_animal/'
edge_input_path = '../../edge_input/'
output_path = '../../output_image/'
csv_output = '../../output_csv_5_25[obvious]/'

_edge_by_channel = ['bgr_gray']

_showImg = { 'original_image':True, 'original_edge':False, 'enhanced_edge':False, 'original_contour':True, 'contour_filtered':True, 'size':False, 'shape':False, 'color':False, 'cluster_histogram':False , 'original_result':True, 'each_obvious_result':True, 'combine_obvious_result':True, 'obvious_histogram':True, 'each_group_result':True, 'result_obvious':True }
_writeImg = { 'original_image':False, 'original_edge':False, 'enhanced_edge':False, 'original_contour':False, 'contour_filtered':False, 'size':False, 'shape':False, 'color':False, 'cluster_histogram':False, 'original_result':False, 'each_obvious_result':False, 'combine_obvious_result':False, 'obvious_histogram':False, 'each_group_result':False, 'result_obvious':False }

_show_resize = [ ( 720, 'height' ), ( 1200, 'width' ) ][0]

test_one_img = { 'test':True , 'filename': 'animal (4).jpg' }
#test_one_img = { 'test':True , 'filename': '13_82.png' }

def main():
     
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
        
        if test_one_img['test'] and i > 0 :
            break
        
        if test_one_img['test']:
            fileName =  test_one_img['filename'] 
        
        print 'Input:',fileName
        
        if not os.path.isfile( input_path + fileName ):
            print input_path + fileName   
            print 'FILE does not exist!'
            break
        
        if not os.path.isfile( edge_input_path + fileName[:-4] + '_edge.jpg' ) and _use_structure_edge:
            print edge_input_path + fileName[:-4] + '_edge.jpg'
            print 'EDGE FILE does not exist!'
            break
        
        # read color image
        color_image_ori = cv2.imread( input_path + fileName )    
    
        height, width = color_image_ori.shape[:2]
        image_resi = cv2.resize( color_image_ori, (0,0), fx= resize_height/height, fy= resize_height/height)        
      
        if _showImg['original_image']:
            cv2.imshow( fileName + ' origianl_image', ShowResize(image_resi) )
            cv2.waitKey(0)
        if _writeImg['original_image']:
            cv2.imwrite(output_path+fileName[:-4]+'_a_original_image.jpg', image_resi )        
        
       
    
        for j in xrange(1):
                     
            if _use_structure_edge :
                            
                # read edge image 
                edge_image_ori = cv2.imread( edge_input_path + fileName[:-4] + '_edge.jpg' , cv2.IMREAD_GRAYSCALE ) 
                height, width = edge_image_ori.shape[:2]
                edged = cv2.resize( edge_image_ori, (0,0), fx= resize_height/height, fy= resize_height/height)                          
                #thresh_gray,edged = cv2.threshold(edge_image_resi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)   
                
            else:
                scale = 1.0/(2**j)
                print 'Scale:', scale           
                str_scale = '1_'+str(2**j)
                image_resi = cv2.resize( image_resi, (0,0), fx= scale, fy= scale)   
                
                if _gaussian_filter :  
                    print 'Gaussian filter'
                    image_resi = cv2.GaussianBlur(image_resi, (gaussian_para, gaussian_para),0)
            
                if _sharpen :
                    print 'Sharpening'
                    image_resi = Sharpen(image_resi)
            
                re_height, re_width = image_resi.shape[:2]
            
                offset_r = re_height/split_n_row
                offset_c = re_width/split_n_column
                
                print 'Detect edge'
                edged = np.zeros(image_resi.shape[:2], np.uint8) 
                
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
                                                   
                        BGR_dic, HSV_dic, LAB_dic = SplitColorChannel( image_resi[ r_l : r_r , c_l : c_r ] )
                                               
                        channel_img_dic = { 'bgr_gray':BGR_dic['img_bgr_gray'], 'b':BGR_dic['img_b'], 'g':BGR_dic['img_g'], 'r':BGR_dic['img_r'], 'h':HSV_dic['img_h'], 's':HSV_dic['img_s'], 'v':HSV_dic['img_v'], 'l':LAB_dic['img_l'], 'a':LAB_dic['img_a'], 'b':LAB_dic['img_b'] }
                        channel_thre_dic = { 'bgr_gray':BGR_dic['thre_bgr_gray'], 'b':BGR_dic['thre_b'], 'g':BGR_dic['thre_g'], 'r':BGR_dic['thre_r'], 'h':HSV_dic['thre_h'], 's':HSV_dic['thre_s'], 'v':HSV_dic['thre_v'], 'l':LAB_dic['thre_l'], 'a':LAB_dic['thre_a'], 'b':LAB_dic['thre_b'] }
                        
                        for chan in _edge_by_channel:
                            if channel_thre_dic[chan] > 20 :
                                edged[ r_l : r_r , c_l : c_r ] = edged[ r_l : r_r , c_l : c_r ] | cv2.Canny( channel_img_dic[chan], 0.5*channel_thre_dic[chan], channel_thre_dic[chan] )
                image_resi = cv2.resize( image_resi, (0,0), fx= 1.0/scale, fy= 1.0/scale)
                edged = cv2.resize( edged, (0,0), fx= 1.0/scale, fy= 1.0/scale)                            
            # end detect edge else  
  
            if _showImg['original_edge']:
                cv2.imshow( fileName + ' origianl_edge', ShowResize(edged) )
                cv2.waitKey(0)
            if _writeImg['original_edge']:
                cv2.imwrite(output_path+fileName[:-4]+'_b_original_edge.jpg', edged )                 
            
            if _enhance_edge and _use_structure_edge:
                # enhance and close the edge
                print 'Enhance edge'
                if _gray_value_redistribution_local : 
                    
                    # create a CLAHE object (Arguments are optional).
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    edged = clahe.apply(edged)                    
            
                    #gray_value_list = range(255,-1,-1)
            
                    #for i in range(len(gray_value_list)):
                        #for a in range(9):
                            #gray_value_list[i] =  int( round( 255 - pow(gray_value_list[i],100)/pow(255.0,99) ) )
            
                    #tmp_height, tmp_width = edge_image_resi.shape[:2]
                    #for h in range(tmp_height) :
                        #for w in range(tmp_width) :
                            #edge_image_resi[h,w] = gray_value_list[ edge_image_resi[h,w] ]     
                else:
                    print 'Equalization'
                    edged = cv2.equalizeHist(edged)            
                
                if _showImg['enhanced_edge']:
                    cv2.imshow( fileName + ' enhanced_edge', ShowResize(edged) )
                    cv2.waitKey(0)
                if _writeImg['enhanced_edge']:
                    cv2.imwrite( output_path + fileName[:-4] +'_c_enhanced_edge.jpg', edged )                 
            # end enhance edge if
                           
            print 'Find countour'     
            edged = cv2.threshold(edged,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            contours = cv2.findContours(edged,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)[-2]
            contour_image = np.zeros( image_resi.shape, np.uint8 )
            color_index = 0 
            
            for c in contours :
                COLOR = switchColor[ color_index % len(switchColor) ]     
                color_index += 1
                cv2.drawContours( contour_image, [c], -1, COLOR, 1 )            
            
            if _showImg['original_contour']:
                cv2.imshow( fileName + ' original_contour', ShowResize(contour_image) )
                cv2.waitKey(0)  
            if _writeImg['original_contour']:
                cv2.imwrite( output_path + fileName[:-4] +'_d_original_contour.jpg', contour_image )                 
                   
            tmp_cnt_list = [contours[0]]
            tmp_cnt = contours[0]
            for c in contours[1:]:
                if not IsOverlap(tmp_cnt,c):
                    tmp_cnt_list.append(c)
                tmp_cnt = c
                
            contours = tmp_cnt_list
            
            noise = 0  
            contour_list = []
            re_height, re_width = image_resi.shape[:2]
            print 'Filter contour'
            print '------------------------'
            for c in contours:
            
                if _remove_small_and_big :
                    # remove too small or too big contour
                    if len(c) < 30 or len(c) > (re_height+re_width)*2/3.0: 
                        continue        
                
                if _checkConvex :
                    # remove contour which is not Convex hull
                    #print 'Check convexhull'
                    if not cv2.isContourConvex(np.array(c)):
                        contour_image = np.zeros( image_resi.shape, np.uint8 )
                        cv2.drawContours( contour_image, [c], -1, GREEN, 1 ) 
                        cv2.imshow( fileName + ' countour', ShowResize(contour_image) )
                        cv2.waitKey(0)                          
                        continue
                
                if _remove_high_density :
                    # remove contour whose density is too large or like a line
                    #print 'Remove contour with high density'
                    area = cv2.contourArea(c) 
                    shape_factor = 4*np.pi*area / float( pow(len(c), 2 ) )
                    if area < 4 or ( len(c) > 30 and float(len(c)) / area > 0.5 ) or ( len(c) <= 30 and float(len(c)) / area > 0.75 ): 
                    #if area < 4 or shape_factor < 0.1 :
                        noise+=1
                        continue
                
                if _remove_too_many_edge :
                    # remove contour which has too many edge
                    #print 'Remove contour with too many edges '
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 10, True)
                    #contour_image = np.zeros( image_resi.shape, np.uint8 )
                    #cv2.drawContours( contour_image, [c], -1, GREEN, 1 )
                    #cv2.imshow('edge number : '+str(len(approx)), ShowResize(contour_image))
                    #cv2.waitKey(0)
                    if len(approx) > 50 : 
                        continue
   
                contour_list.append(c)
            # end filter contour for
            
            contour_dic_list = []
            for cnt in contour_list:
                contour_dic_list.append( {'cnt':cnt} )
            # remove outer contour of two overlapping contours whose sizes are close          
            if _check_overlap :     
                print 'Remove overlap contour keep inner ones'
                contour_dic_list = CheckOverlap(contour_dic_list)
            
            contour_list = []
            for cnt_dic in contour_dic_list:
                contour_list.append(cnt_dic['cnt'])
                
            if len(contour_list) == 0 :
                continue
            
            print '------------------------'
                       
            
            # draw contour by different color
            contour_image = np.zeros( image_resi.shape, np.uint8 )
            contour_image[:] = BLACK
            color_index = 0 
            for c in contour_list :
                COLOR = switchColor[ color_index % len(switchColor) ]     
                color_index += 1
                cv2.drawContours( contour_image, [c], -1, COLOR, 1 )
                #cv2.drawContours( gradient_img_copy, [c], -1, COLOR, 1 )
            
            #combine_image = np.concatenate( (gradient_img, gradient_img_copy), axis = 1)
            
            #cv2.imwrite( './gradient.jpg',combine_image)            
            #cv2.imshow( 'gradient',combine_image)
            #cv2.waitKey(0)
    
            if _showImg['contour_filtered']:
                cv2.imshow( fileName + ' contour_filtered', ShowResize(contour_image) )
                cv2.waitKey(0)
            if _writeImg['contour_filtered']:
                cv2.imwrite( output_path + fileName[:-4] +'_e_contour_filtered.jpg', contour_image )

            print 'Extract contour feature'
            # Get contour feature
            c_list, cnt_shape_list, cnt_color_list, cnt_size_list, cnt_color_gradient_list = get_contour_feature.extract_feature( image_resi, contour_list )

            cnt_dic_list = []
            for i in range( len(c_list) ):
                cnt_dic_list.append( { 'cnt':c_list[i], 'shape':cnt_shape_list[i], 'color':cnt_color_list[i], 'size':cnt_size_list[i], 'color_gradient':cnt_color_gradient_list[i] } )
            
            feature_dic = { 'cnt':c_list, 'shape':cnt_shape_list, 'color':cnt_color_list, 'size':cnt_size_list }
        
            para = [ 'size', 'shape' , 'color' ] 
            
            # total contour number
            cnt_N = len(c_list)
            
            label_list_dic = {}
            
            print 'Respectively use shape, color, and size as feature set to cluster'
            # Respectively use shape, color, and size as feature set to cluster
            for para_index in xrange( len(para) ):
                
                print 'para:',para[para_index]
   
                contour_feature_list =  feature_dic[para[para_index]]
                
                # hierarchical clustering
                label_list = Hierarchical_clustering( contour_feature_list, fileName, para[para_index] )   
                
                unique_label, label_counts = np.unique(label_list, return_counts=True)
                
                # draw contours of each group refer to the result clustered by size, shape or color
                contour_image = np.zeros(image_resi.shape, np.uint8)
                contour_image[:] = BLACK                      
                color_index = 0    
                for label in unique_label :
                    COLOR = switchColor[ color_index % len(switchColor) ]
                    color_index += 1
                    tmp_splited_group = []
                    for i in xrange( len(label_list) ):
                        if label_list[i] == label :
                            tmp_splited_group.append( c_list[i] )                        
                    cv2.drawContours( contour_image, np.array(tmp_splited_group), -1, COLOR, 2 )
                
                if _showImg[para[para_index]]:
                    cv2.imshow( 'cluster by :'+ para[para_index], ShowResize(contour_image) )
                    cv2.waitKey(0)
                if _writeImg[para[para_index]]:
                    cv2.imwrite( output_path + fileName[:-4] +'_f_para['+para[para_index]+'].jpg', contour_image ) 
                
                label_list_dic[para[para_index]] = label_list
                
            # end para_index for
            
            # intersect the label clustered by size, shpae, and color
            combine_label_list = []
            for i in xrange( cnt_N ):
                combine_label_list.append( str(label_list_dic['size'][i]) + '_' + str(label_list_dic['shape'][i]) + '_' + str(label_list_dic['color'][i])  )
                
            unique_label, label_counts = np.unique(combine_label_list, return_counts=True)      
            label_dic = dict(zip(unique_label, label_counts))
            max_label = max( label_dic.iteritems(), key=operator.itemgetter(1) )[0]
       
            # find the final group by the intersected label and draw
            final_group = []  
            contour_image = np.zeros(image_resi.shape, np.uint8)
            contour_image[:] = BLACK     
            contour_image_max = np.zeros(image_resi.shape, np.uint8)
            contour_image_max[:] = BLACK 
            
                        
            color_index = 0             
            for label in unique_label :
                contour_image_each = image_resi.copy()
                # darken the image to make the contour visible
                contour_image_each[:] = contour_image_each[:]/3.0
                COLOR = switchColor[ color_index % len(switchColor) ]
                color_index += 1
                tmp_group = []
                for i in xrange( cnt_N ):
                    if combine_label_list[i] == label :
                        tmp_group.append( cnt_dic_list[i] ) 
                
                tmp_group = CheckOverlap(tmp_group)
                tmp_cnt_group = []
                avg_color_gradient = 0.0
                avg_shape_factor = 0.0
                tmp_area = 0.0
                
                #for each final group count factor
                for cnt_dic in tmp_group:
                    cnt = cnt_dic['cnt']
                    cnt_area = cv2.contourArea(cnt)
                    tmp_area += cnt_area
                    avg_shape_factor += (4*np.pi*cnt_area)/float(pow(len(cnt),2))
                    avg_color_gradient += cnt_dic['color_gradient']
                    tmp_cnt_group.append(cnt)
                
                avg_shape_factor /= float(len(tmp_group))
                avg_color_gradient /= float(len(tmp_group))
                
                if len(tmp_cnt_group) < 2 :
                    continue
                
                if label == max_label :                  
                    cv2.drawContours( contour_image_max, np.array(tmp_cnt_group), -1, RED, 2 )
                else:
                    cv2.drawContours( contour_image_max, np.array(tmp_cnt_group), -1, GREEN, 1 ) 
            
                cv2.drawContours( contour_image, np.array(tmp_cnt_group), -1, COLOR, 2 )
                cv2.drawContours( contour_image_each, np.array(tmp_cnt_group), -1, COLOR, 2 )
                
                
                final_group.append( { 'cnt':tmp_cnt_group, 'cover_area':tmp_area, 'color_gradient':avg_color_gradient, 'shape_factor':avg_shape_factor, 'obvious_weight':0, 'combine_weight':0.0 } )
                
                contour_image_each = cv2.resize( contour_image_each, (0,0), fx = float(color_image_ori.shape[0])/contour_image_each.shape[0], fy = float(color_image_ori.shape[0])/contour_image_each.shape[0])
                
            # end find final group for
            # sort the group from the max area to min group and get max count
           
            if _showImg['original_result']:
                cv2.imshow(fileName+' original_result', ShowResize(contour_image) )
                cv2.waitKey(0)     
            if _writeImg['original_result']:
                cv2.imwrite( output_path + fileName[:-4] +'_g_original_result.jpg', contour_image )            
                
            if len(final_group) < 1:
                print 'No any pattern'
                continue
            
            obvious_list = ['cover_area','color_gradient','shape_factor']
            #sort final cnt group by cover_area , shape_factor and color_gradient
            for obvious_para in obvious_list:
                
                if obvious_para == 'color_gradient':
                    avg_img_gradient = Avg_Img_Gredient(image_resi)
                    final_group.append( { 'cnt':[], 'cover_area':[], 'color_gradient':avg_img_gradient, 'shape_factor':[], 'obvious_weight':-1, 'combine_weight':-1 } )
                    print 'avg_img_gradient:',avg_img_gradient
                    
                final_group.sort( key = lambda x:x[obvious_para], reverse = True )
                obvious_index = len(final_group)-1
                max_diff = 0
                area_list = [ final_group[0][obvious_para] ]
                
               
                final_group[0]['combine_weight'] += 1.0
                
                for i in range( 1, len( final_group ) ):
                    area_list.append(final_group[i][obvious_para])
                    diff = final_group[i-1][obvious_para] - final_group[i][obvious_para]
                    if final_group[i]['combine_weight'] != -1 :
                        final_group[i]['combine_weight'] += final_group[i][obvious_para]/float(final_group[0][obvious_para])
                    
                    if 0.8*final_group[i-1][obvious_para] > final_group[i][obvious_para] and diff > max_diff:
                        max_diff = diff
                        obvious_index = i-1
                   
                print obvious_para,'_list:',area_list
                
                contour_image[:] = BLACK
                for i in range(obvious_index+1):
                    if final_group[i]['obvious_weight'] == -1:
                        obvious_index = i
                        break
                    final_group[i]['obvious_weight'] += 1
                    cv2.drawContours( contour_image, np.array(final_group[i]['cnt']), -1, GREEN, 2 )
                for i in range(obvious_index+1,len(final_group)):
                    cv2.drawContours( contour_image, np.array(final_group[i]['cnt']), -1, RED, 2 )  
                    
                if _showImg['each_obvious_result']:
                    cv2.imshow(fileName+' obvious_para:['+obvious_para+'] | Green for obvious', ShowResize(contour_image) )
                    cv2.waitKey(0)     
                if _writeImg['each_obvious_result']:
                    cv2.imwrite( output_path + fileName[:-4] +'_h_para['+obvious_para+']_obvious(Green).jpg', contour_image )   
                
                plt.bar(left=range(len(area_list)),height=area_list)   
                plt.title( obvious_para+' cut_point : '+str(obvious_index)+'  | value: '+str(final_group[obvious_index][obvious_para]) )
                       
                if _showImg['obvious_histogram']:
                    plt.show()
                if _writeImg['obvious_histogram']:
                    plt.savefig(output_path+fileName[:-4]+'_h_para['+obvious_para+']_obvious_his.png')    
                plt.close()  
                
                if obvious_para == 'color_gradient':
                    final_group.remove({ 'cnt':[], 'cover_area':[], 'color_gradient':avg_img_gradient, 'shape_factor':[], 'obvious_weight':-1, 'combine_weight':-1 })
                
            # end obvious para for
            
            final_obvious_group = []
            if _use_comebine_weight :
                final_group.sort( key = lambda x:x['combine_weight'], reverse = True )
                obvious_index = len(final_group)-1
                max_diff = 0
                area_list = [ final_group[0]['combine_weight'] ]
                
                for i in range( 1, len( final_group ) ):
                    area_list.append(final_group[i]['combine_weight'])
                    diff = final_group[i-1]['combine_weight'] - final_group[i]['combine_weight']
                    
                    if final_group[i-1]['combine_weight'] > final_group[i]['combine_weight'] and diff > max_diff:
                        max_diff = diff
                        obvious_index = i-1
                   
                print obvious_para,'_list:',area_list  
                
                for i in range(obvious_index+1):
                    final_obvious_group.append(final_group[i])
                    cv2.drawContours( contour_image, np.array(final_group[i]['cnt']), -1, GREEN, 2 )
                for i in range(obvious_index+1,len(final_group)):
                    cv2.drawContours( contour_image, np.array(final_group[i]['cnt']), -1, RED, 2 )  
                    
               
                if _showImg['combine_obvious_result']:
                    cv2.imshow(fileName+' combine_obvious_result | Green for obvious', ShowResize(contour_image) )
                    cv2.waitKey(0)     
                if _writeImg['combine_obvious_result']:
                    cv2.imwrite( output_path + fileName[:-4] +'_combine_obvious_result(Green).jpg', contour_image )   
                
                plt.bar(left=range(len(area_list)),height=area_list)   
                plt.title( 'combine_obvious_result cut_point : '+str(obvious_index)+'  | value: '+str(final_group[obvious_index][obvious_para]) )
                       
                if _showImg['obvious_histogram']:
                    plt.show()
                if _writeImg['obvious_histogram']:
                    plt.savefig(output_path+fileName[:-4]+'_combine_obvious_result_his.png')    
                plt.close()                  
                
            else:
                weight = 2
                final_group.sort( key = lambda x:x['obvious_weight'], reverse = True )
                if final_group[0]['obvious_weight'] == 3:
                    weight = 3
                for f_group in final_group :
                    # determine obvious if match more than two obvious condition 
                    if f_group['obvious_weight'] == weight:
                        final_obvious_group.append(f_group)
                        
            # end choose obvious way if 
            
            final_group_cnt = []
            contour_image[:] = BLACK
            for tmp_group in final_obvious_group:
                tmp_group = tmp_group['cnt']
                
                if len(tmp_group) < 2 :
                    continue
                
                final_group_cnt.append(tmp_group)
                contour_image_each = image_resi.copy()
                # darken the image to make the contour visible
                contour_image_each[:] = contour_image_each[:]/3.0                
                COLOR = switchColor[ color_index % len(switchColor) ]
                color_index += 1                
                cv2.drawContours( contour_image, np.array(tmp_group), -1, COLOR, 2 )
                cv2.drawContours( contour_image_each, np.array(tmp_group), -1, COLOR, 2 )
    
                if _showImg['each_group_result']:            
                        cv2.imshow(fileName+' each_group_result', ShowResize(contour_image_each) )
                        cv2.waitKey(0)     
                if _writeImg['each_group_result']:
                        cv2.imwrite( output_path + fileName[:-4] +'_i_label['+str(color_index)+']_Count['+str(len(tmp_group))+'].jpg', contour_image_each )  
    
    
            contour_image = cv2.resize( contour_image, (0,0), fx = height/resize_height, fy = height/resize_height)
            combine_image = np.concatenate((color_image_ori, contour_image), axis=1) 
    
            if _showImg['result_obvious']:
                cv2.imshow(fileName+' result_obvious', ShowResize(combine_image) )
                cv2.waitKey(0)     
            if _writeImg['result_obvious']:
                cv2.imwrite( output_path + fileName[:-4] +'_j_result_obvious.jpg', combine_image )            
    
    
            if _record_by_csv:
                Record_by_CSV( fileName, final_group_cnt, contour_image )

           
        #end scale for
        print 'Finished in ',time.time()-start_time,' s'
        #for img_tuple in each_scale_best_result:
            #print '--------------------------------------------------'
            #print img_tuple['filename']
            ##print 'Total Count : ',img_tuple['count']
            ##cv2.imwrite( output_path + img_tuple['filename'] +'_Clu['+str(img_tuple['evaluate_score'])+']_Count['+str(img_tuple['count'])+']'+'.jpg', img_tuple['img'] ) 
            #cv2.imshow(img_tuple['filename'], cv2.resize( img_tuple['img'], (0,0), fx= resize_ratio, fy= resize_ratio))
        #cv2.waitKey(0)
        
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
        

def Avg_Img_Gredient( img, model = 'lab' ):
    
    kernel = np.array( [[-1,-1,-1],
                        [-1, 8,-1],
                        [-1,-1,-1]]  )    
    
    if model == 'lab' :
        
        height, width = img.shape[:2]       
        lab = cv2.cvtColor( img, cv2.COLOR_BGR2LAB)
        lab_l = lab[:,:,0]
        lab_a = lab[:,:,1]
        lab_b = lab[:,:,2]      
     
        lab_list = [ lab_l, lab_a, lab_b ]
        gradient_list = []
        
        for lab_channel in lab_list :
            gradient = cv2.filter2D(lab_channel, -1, kernel)
            gradient_list.append(gradient)
        
        avg_gradient = 0.0  
        for x in range(height):
            for y in range(width):
                avg_gradient += math.sqrt( pow( gradient_list[0][x,y],2 ) +  pow( gradient_list[1][x,y],2 ) + pow( gradient_list[2][x,y],2 ) )  
                
        avg_gradient /=  ( float(height) * float(width) )
                
    return avg_gradient       
                

def Record_by_CSV( filename, cnt_list, contour_image ):
    
    coordinar_list = [ [ 'Group','Y','X' ] ]
    img = contour_image.copy()
    #img[:]=BLACK
    # for each group
    for group_i in range( len(cnt_list) ):
        for cnt in cnt_list[group_i]:
            x, y = GetMoment(cnt)
            coordinar_list.append( [ group_i, int(y), int(x) ] )
            cv2.circle(img,(int(y),int(x)),2,(0,0,255),2)
    #cv2.imshow('coordinate: '+str(x)+','+str(y),img)
    #cv2.waitKey(0)
    f = open(csv_output+filename[:-4]+'.csv',"wb")
    w = csv.writer(f)
    w.writerows(coordinar_list)
    f.close()       
    
def Sharpen(img):
    
    kernel_sharpen = np.array([[-1,-1,-1,-1,-1],
                                 [-1,2,2,2,-1],
                                 [-1,2,8,2,-1],
                                 [-1,2,2,2,-1],
                                 [-1,-1,-1,-1,-1]]) / 8.0  
   
    return cv2.filter2D(img, -1, kernel_sharpen) 

def Eucl_distance(a,b):
    
    if type(a) != np.ndarray :
        a = np.array(a)
    if type(b) != np.ndarray :
        b = np.array(b)
    
    return np.linalg.norm(a-b) 

def Draw_image( image_resi, c_list, label_list, max_label ):
    
    if type(label_list) != np.ndarray :
        label_list = np.array(label_list)
    
    samples_mask = np.zeros_like(c_list, dtype=bool)
    samples_mask[:] = True     
    index_mask = ( label_list == max_label )
    image = image_resi.copy()
    contour_image = np.zeros(image_resi.shape, np.uint8)
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

def CheckOverlap( cnt_dic_list ):
    
    if cnt_dic_list == []:
        return []
    
    checked_list = []
    # sort list from little to large
    cnt_dic_list.sort( key = lambda x: len(x['cnt']) , reverse = False)
    
    for c_dic in cnt_dic_list  : 
        if IsOverlapAll( c_dic, checked_list ) : 
            continue               
        checked_list.append(c_dic) 
    
    return checked_list
        
def IsOverlap( cnt1, cnt2 ):
    
    if cnt1 == [] or cnt2 == [] :
        return False
    
    c1M = GetMoment(cnt1)
    c2M = GetMoment(cnt2)
    c1_min_d = MinDistance(cnt1)
    c2_min_d = MinDistance(cnt2)
    moment_d = Eucl_distance( c1M, c2M )
    
    if min(c1_min_d,c2_min_d) == 0:
        return False
    
    return ( moment_d < c1_min_d or moment_d < c2_min_d ) and max(c1_min_d,c2_min_d)/min(c1_min_d,c2_min_d) <= 3

def IsOverlapAll( cnt_dic, cnt_dic_list ):
    
    if cnt_dic == [] or len(cnt_dic_list) < 1 :
        return False

    for c_dic in cnt_dic_list :
        #if len(c) == len(cnt) and GetMoment(c) == GetMoment(cnt):
            ##print 'same one'
            #continue
        if IsOverlap( cnt_dic['cnt'], c_dic['cnt'] ) :
            return True
    
    return False

def SplitColorChannel( img ):
    
    bgr_gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY ) 
    bgr_gray = cv2.GaussianBlur(bgr_gray, (5, 5), 0)  
    thresh_bgr_gray = cv2.threshold(bgr_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]    
    
    
    bgr_b = img[:,:,0]
    bgr_g = img[:,:,1]
    bgr_r = img[:,:,2]   
    bgr_b = cv2.GaussianBlur(bgr_b, (5, 5), 0)
    bgr_g = cv2.GaussianBlur(bgr_g, (5, 5), 0)
    bgr_r = cv2.GaussianBlur(bgr_r, (5, 5), 0)
    thresh_bgr_b = cv2.threshold(bgr_b,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    thresh_bgr_g = cv2.threshold(bgr_g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    thresh_bgr_r = cv2.threshold(bgr_r,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]    
    
    hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
    hsv_h = hsv[:,:,0]
    hsv_s = hsv[:,:,1]
    hsv_v = hsv[:,:,2]   
    hsv_h = cv2.GaussianBlur(hsv_h, (5, 5), 0)
    hsv_s = cv2.GaussianBlur(hsv_s, (5, 5), 0)
    hsv_v = cv2.GaussianBlur(hsv_v, (5, 5), 0)
    thresh_hsv_h = cv2.threshold(hsv_h,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    thresh_hsv_s = cv2.threshold(hsv_s,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    thresh_hsv_v = cv2.threshold(hsv_v,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]                           

    lab = cv2.cvtColor( img, cv2.COLOR_BGR2LAB)
    lab = cv2.GaussianBlur(lab, (5,5), 0)
    lab_l = lab[:,:,0]
    lab_a = lab[:,:,1]
    lab_b = lab[:,:,2]
    lab_l = cv2.GaussianBlur(lab_l, (5, 5), 0)
    lab_a = cv2.GaussianBlur(lab_a, (5, 5), 0)
    lab_b = cv2.GaussianBlur(lab_b, (5, 5), 0)  
    thresh_lab_l = cv2.threshold(lab_l,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    thresh_lab_a = cv2.threshold(lab_a,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    thresh_lab_b = cv2.threshold(lab_b,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]   
    
    return { 'img_bgr_gray':bgr_gray, 'img_bgr':img, 'img_b':bgr_b, 'img_g':bgr_g, 'img_r':bgr_r, 'thre_bgr_gray':thresh_bgr_gray, 'thre_b':thresh_bgr_b, 'thre_g':thresh_bgr_g, 'thre_r':thresh_bgr_r 
             },{ 'img_hsv':hsv, 'img_h':hsv_h, 'img_s':hsv_s, 'img_v':hsv_v, 'thre_h':thresh_hsv_h, 'thre_s':thresh_hsv_s, 'thre_v':thresh_hsv_v 
                 },{ 'img_lab':lab, 'img_l':lab_l, 'img_a':lab_a, 'img_b':lab_b, 'thre_l':thresh_lab_l, 'thre_a':thresh_lab_a, 'thre_b':thresh_lab_b }

def ShowResize( img ):
    
    h, w = img.shape[:2]
    
    if _show_resize[1] == 'height':
        ratio = _show_resize[0] / float(h)
    else :
        ratio = _show_resize[0] / float(w)
    
    return cv2.resize( img, (0,0), fx = ratio, fy = ratio )
        
def MinDistance(cnt):
    
    cM = GetMoment(cnt)
    min_d = Eucl_distance( (cnt[0][0][0],cnt[0][0][1]), cM )
    for c in cnt :
        d = Eucl_distance( (c[0][0],c[0][1]), cM ) 
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
        cx += float(c[0][0])
        cy += float(c[0][1])
        
    return float(cx)/num, float(cy)/num

def Hierarchical_clustering( feature_list, fileName, para, cut_method = 'elbow' ):

    if len(feature_list) < 2:
        return [0]*len(feature_list)
    
    # hierarchically link cnt by order of distance from distance method 'ward'
    #print feature_list
    cnt_hierarchy = linkage( feature_list, 'ward')
    #cnt_hierarchy = linkage( feature_list)
    
    max_d = 10
    if cut_method == 'elbow' or True:
        last = cnt_hierarchy[:, 2]
        #print 'last:',last
        last = [ x for x in last if x > 0 ]
        #print 'last:',last
        acceleration = np.diff(last) 
        
        #acceleration = map(abs, np.diff(acceleration) )
        
        #acceleration_rev = acceleration[::-1]
        #print 'acceleration:',acceleration 
        
        if len(acceleration) < 2 :
            return [0]*len(feature_list)
        avg_diff = sum(acceleration)/float(len(acceleration))
        tmp = acceleration[0]
        
        avg_list = [x for x in acceleration if x > avg_diff]
        avg_diff = sum(avg_list)/float(len(avg_list))
        
        off_set = 5
        
        cut_point_list = []
        for i in xrange( 1,len(acceleration) ):
       
            if acceleration[i] > avg_diff:
                #cut_point_list.append( [ i, acceleration[i]/(tmp/float(i) ) ] )
                
                tmp_offset_prev = off_set
                prev = i - off_set
                if prev < 0 :
                    prev = 0
                    tmp_offset_prev = i-prev
                
                cut_point_list.append( [ i, acceleration[i]/( sum(acceleration[prev:i]) / float(tmp_offset_prev) ) ] )
                #print 'i:',i+1,' ratio:',acceleration[i]/( sum(acceleration[n:i]) / float(off_set) )
                
            tmp += acceleration[i]
            
        if len(cut_point_list) < 1 :
            print 'all in one group!'
            return [0]*len(feature_list)     
        
        cut_point_list.sort( key = lambda x : x[1], reverse = True )
        
        #print 'cut index:',cut_point_list[0][0]+1,' diff len:',len(acceleration)
        max_d = last[cut_point_list[0][0]]
        max_ratio = cut_point_list[0][1]
        
        if max_ratio < 2.0 :
            print 'all in one group! max_ratio:',max_ratio
            return [0]*len(feature_list)  
        
        #max_d = last[acceleration.argmax()]
    #elif cut_method == 'inconsistency':
    
    plt.bar(left=range(len(acceleration)),height=acceleration)   
    plt.title( para+' cut_point : '+str(cut_point_list[0][0]+1)+'  | value: '+str(acceleration[cut_point_list[0][0]])+' | ratio: '+ str(max_ratio)  )
    
    if _showImg['cluster_histogram']:
        plt.show()
    if _writeImg['cluster_histogram']:
        plt.savefig(output_path+fileName[:-4]+'_f_para['+para+']_his.png')    
    plt.close()    
    
    #print 'acceleration.argmax():',acceleration.argmax()
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
