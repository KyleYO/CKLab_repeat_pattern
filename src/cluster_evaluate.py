import numpy as np
import math

def getMoment( group_a ):
    
    if type(group_a[0]) != np.ndarray and type(group_a[0]) != list :
        for i in xrange(len(group_a)):
            group_a[i] = [group_a[i]]
    
    number = len(group_a)
    if group_a == []:
        return group_a
    
    Moment = group_a[0]
    
    if number < 2 :
        return Moment
    
    for c in group_a[1:]:
        Moment = map( sum, zip(Moment,c) )
        
    for i in xrange(len(Moment)):
        Moment[i] = float(Moment[i]) / number
        
    return Moment

def minDistance( group_a, group_b ):
    
    minDis = Eucl_distance( group_a[0], group_b[0] )
    
    for cnt_a in group_a:
        for cnt_b in group_b:
            tmp = Eucl_distance( cnt_a, cnt_b )
            if tmp < minDis :
                minDis = tmp
          
            
    return minDis
    
def Eucl_distance(a,b):
    
    if type(a) != np.ndarray :
        a = np.array(a)
    if type(b) != np.ndarray :
        b = np.array(b)
    
    return np.linalg.norm(a-b)  

def Discrimination_rate( group_a, group_b ):
    
    Moment_dis = Eucl_distance( getMoment( group_a ), getMoment( group_b ) )
    if Moment_dis == 0:
        #print 'Moment_dis == 0'
        return 0
    dimention = len(group_a[0])
    
    min_dis = minDistance( group_a, group_b )
    
    #if min_dis >= Moment_dis :
        #print 'min_dis >= Moment_dis'
        #return 0
    #return  float(min_dis)/ Moment_dis
    return  float(min_dis)
    
    # / math.sqrt(dimention) for nomalize to [0-1]
    #return float( min_dis)/ math.sqrt(dimention)

def minDiscrimination_rate( group_a, group_list ):
    
    if len( group_list ) < 2 :
        return 0 
    
    minDiscrim = 10
    for group in group_list :
        tmp = Discrimination_rate( group, group_a )
        if tmp != 0 and tmp < minDiscrim :
            minDiscrim = tmp
            
    #if minDiscrim > 1:
        #print 'minDiscrim:',minDiscrim
        #return 0
    
    return minDiscrim

def minDiscrimination_rate_All( group_list ):
    
    if len( group_list ) < 2 :
            return 1.0/len(group_list[0])
        
    minDiscrim = Discrimination_rate( group_list[0], group_list[1] )
    for i in xrange( len(group_list)-1 ) :
        for j in xrange( i+1, len(group_list) ) :          
            tmp = Discrimination_rate( group_list[i], group_list[j] )
            if tmp != 0 and tmp < minDiscrim :
                minDiscrim = tmp
                
            
    #if minDiscrim > 1:
        #print 'minDiscrim:',minDiscrim
        #return 0
    
    return minDiscrim    

def avgDistance( group_a ):
    
    if len( group_a ) < 2 :
        return 0
    
    avgDis = 0.0
    vector_number = len(group_a)
    moment = getMoment(group_a)
    
    for contour in group_a :
        avgDis += Eucl_distance( contour, moment)
    
    return avgDis/vector_number

def Agglomerate_rate( group_a ):
    
    if len(group_a) < 2 :
        return 1
    
    std_devia = Standard_deviation( group_a ) 
    moment = getMoment(group_a)
    avgDis = avgDistance(group_a)
    avgDis_3std = avgDis + 3*std_devia
    avgDis_N = 0
    avgDis_3std_N = 0
    
    for contour in group_a :
        if Eucl_distance( moment, contour ) <= avgDis : 
            avgDis_N += 1
        if Eucl_distance( moment, contour ) <= avgDis_3std :
            avgDis_3std_N += 1
    
    return float(avgDis_N)/avgDis_3std_N

def Agglomerate_rate_All( group_list ):
    
    avg = 0.0
    for group_a in group_list:
        avg += Agglomerate_rate( group_a )
        #avg += Standard_deviation( group_a )
    
    return float(avg)/len(group_list)

def Standard_deviation( group_a ):
    
    Moment = getMoment(group_a)
    total = 0.0   
    for vector in group_a:
        total += math.pow(Eucl_distance( vector, Moment ), 2) 
    
    return math.sqrt( float(total)/len(group_a) )   

def AvgGroupMomentDistance( group_list ):
    
    avg_dis = 0.0
    group_n = len(group_list)
    
    removed_outlier_list = []
    for group_a in group_list:
        if len(group_a) < 2 :
            continue
        removed_outlier_list.append(group_a)
    group_n = len(removed_outlier_list)
    
    if group_n < 1:
        return 0
    
    for group_a in removed_outlier_list :
        avg_dis += avgDistance(group_a)
    
    return float(avg_dis)/group_n
    
def Density_rate( group_a, group_list ):
    
    group_N = len( group_a )
    dimention = len( group_a[0] )

    if group_N < 2 :
        return 0
    
    total_N = 0
    
    max_list = [0] * dimention 
    min_list = [100] * dimention 
    
    for group in group_list : 
        total_N += len(group)
        for contour in group :
            for i in xrange( len(contour) ) :
                if contour[i] > max_list[i] :
                    max_list[i] = contour[i]
                if contour[i] < min_list[i] :
                    min_list[i] = contour[i]
                 
    total_space = np.array( max_list ) - np.array(  min_list )
       
        
    max_list = [0] * dimention
    min_list = [100] * dimention 
    
    for contour in group_a :
        for i in xrange( len(contour) ) :
                if contour[i] > max_list[i] :
                    max_list[i] = contour[i]
                if contour[i] < min_list[i] :
                    min_list[i] = contour[i]
   
    group_space = np.array(max_list) - np.array(min_list)
    
    contour_ratio = float(total_N)/group_N
    space_ratio = 1.0
    
    for i in xrange( len( group_space ) ):
        #print space_ratio
        space_ratio = round( (space_ratio*float(total_space[i])/group_space[i]), 2)
        #print space_ratio,float(total_space[i]),group_space[i]
    
    #print contour_ratio,space_ratio
    #return 1 - ( contour_ratio/space_ratio)
    return 1/contour_ratio

# evalute max group to all group
def Cluster_evaluation( group_a, group_list ):
    
    #return Density_rate( group_a, group_list ) , Agglomerate_rate( group_a ) , minDiscrimination_rate( group_a, group_list )
    return Agglomerate_rate( group_a ) , minDiscrimination_rate( group_a, group_list )

# evalute all group average
def Cluster_evaluation_All( group_list ):
    
    if len(group_list) < 1:
        return 0, 0
    #remove_outlier = []
    #for group in group_list:
        #if len(group) > 1 :
            #remove_outlier.append(group)
    
    #if len(remove_outlier) == 0:
        #return 0, minDiscrimination_rate_All( group_list )
    
    return Agglomerate_rate_All( group_list ) , minDiscrimination_rate_All( group_list )
    #return 1-AvgGroupMomentDistance( group_list ) , minDiscrimination_rate_All( group_list )
    