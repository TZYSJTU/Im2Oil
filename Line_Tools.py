# tools for line drawing

import math
import numpy as np
import cv2 

def IntRound(x):
    return int(round(x))

def FindPath_V0(img_gray, angle,y,x):
    path = [[y,x]]
    
    (H,W) = angle.shape
    begin_x = x
    begin_y = y
    #  angle_begin =  angle[int(round(begin_y)),int(round(begin_x))]
    while(1):
        angle_temp = angle[int(round(begin_y)),int(round(begin_x))]
        if abs(angle_temp) > 45:
            delta_y = 1 * np.sign(angle_temp)
            delta_x = delta_y/math.tan(angle_temp/180*math.pi)

        if abs(angle_temp)<45 or abs(angle_temp)==45:
            delta_x = 1
            delta_y = math.tan(angle_temp/180*math.pi)

        begin_x += delta_x
        begin_y -= delta_y

        int_x = int(round(begin_x))
        int_y = int(round(begin_y))
        if int_y<0 or int_y> H-1 or int_x> W-1 or int_x<0:
            break

        angle_next = angle[int_y, int_x]

        if len(path)<40: 
            path.append([int_y,int_x])
        else:
            break

            
    begin_x = x
    begin_y = y
    while(1):
        angle_temp = angle[int(round(begin_y)),int(round(begin_x))]
        if abs(angle_temp) > 45:
            delta_y = -1 * np.sign(angle_temp)
            delta_x = -delta_y/math.tan(angle_temp/180*math.pi)

        if abs(angle_temp)<45 or abs(angle_temp)==45:
            delta_x = -1
            delta_y = -math.tan(angle_temp/180*math.pi)

        begin_x += delta_x
        begin_y -= delta_y

        int_x = int(round(begin_x))
        int_y = int(round(begin_y))
        if int_y<0 or int_y> H-1 or int_x> W-1 or int_x<0:
            break

        angle_next = angle[int_y, int_x]

        # a1 = img[int_y, int_x,2]
        # a2 = img[y, x,2]
        if len(path)<40: 
            path.append([int_y,int_x])
        else:
            break



    return path


# FindPath_V1() 根据ETF向量场搜索，有最短长度，根据灰度有终止条件
# graymap 灰度图, angle 向量场, y, x, step_length 步长, min_length 线的最短步长
# 圆心位置是亚像素的      ETF方向不用亚像素，否则太复杂了
def FindPath_V1(graymap, angle, y, x, step_length = 1, min_length = 20, max_length=60, threshold = 30):
    (H,W) = graymap.shape       # image shape
    path = [[y,x]]              # path: a list to save points' trace [y,x]
    
    # grayscale = graymap[IntRound(y),IntRound(x)]
    grayscale = 0

    # direction positive and negative 
    positive_x, positive_y = x, y       # begin point for positive direction
    negative_x, negative_y = x, y       # begin point for negative direction
    Flag_positive = True                # stop condition
    Flag_negative = True                # stop condition
    while(1):
        if Flag_positive == True:
            positive_angle = angle[IntRound(positive_y), IntRound(positive_x)]/180*math.pi
            positive_gray = graymap[IntRound(positive_y), IntRound(positive_x)]
            positive_delta_y = step_length * math.sin(positive_angle)
            positive_delta_x = step_length * math.cos(positive_angle)
            positive_y -= positive_delta_y       # -=
            positive_x += positive_delta_x       # +=
            

        if Flag_negative == True:
            negative_angle = angle[IntRound(negative_y), IntRound(negative_x)]/180*math.pi
            negative_gray = graymap[IntRound(negative_y), IntRound(negative_x)]
            negative_delta_y = -step_length * math.sin(negative_angle)
            negative_delta_x = -step_length * math.cos(negative_angle)
            negative_y -= negative_delta_y       # -=
            negative_x += negative_delta_x       # +=
            

        # if out of canvas, stop
        if positive_y<0 or IntRound(positive_y)> H-1 or IntRound(positive_x)> W-1 or positive_x<0:
            Flag_positive = False
        if negative_y<0 or IntRound(negative_y)> H-1 or IntRound(negative_x)> W-1 or negative_x<0:
            Flag_negative = False



       
        

        if Flag_positive == True:
            grayscale_positive = graymap[IntRound(positive_y),IntRound(positive_x)]
            if len(path) < min_length: 
                path.append([positive_y, positive_x])
            elif abs(grayscale_positive-grayscale)<threshold:
                path.append([positive_y, positive_x])
            else:
                Flag_positive = False     

        if Flag_negative == True:
            grayscale_negative = graymap[IntRound(negative_y),IntRound(negative_x)]
            if len(path) < min_length: 
                path.insert(0,[negative_y, negative_x])
            elif abs(grayscale_negative-grayscale)<threshold:
                path.insert(0,[negative_y, negative_x])
            else:
                Flag_negative = False  
        
        if Flag_positive == False and Flag_negative == False:
            break

        if len(path)> max_length:
            break

    return path


# FindPath_V2() 根据ETF向量场搜索曲线，有最短长度，灰度差取小于的像素
# graymap 灰度图, angle 向量场, y, x, step_length 步长, min_length 线的最短步长
# 圆心位置是亚像素的      ETF方向不用亚像素，否则太复杂了
def FindPath_V2(graymap, angle, y, x, step_length = 1, min_length = 40, max_length=80, threshold = 25):

    (H,W) = graymap.shape       # image shape
    path = [(y,x)]              # path: a list to save points' trace [y,x]
    
    grayscale = graymap[IntRound(y),IntRound(x)]
    # grayscale = 0

    # direction positive and negative 
    positive_x, positive_y = x, y       # begin point for positive direction
    negative_x, negative_y = x, y       # begin point for negative direction
    Flag_positive = True                # stop condition
    Flag_negative = True                # stop condition
    while(1):
        if Flag_positive == True:
            positive_angle = angle[IntRound(positive_y), IntRound(positive_x)]/180*math.pi
            positive_gray = graymap[IntRound(positive_y), IntRound(positive_x)]
            positive_delta_y = step_length * math.sin(positive_angle)
            positive_delta_x = step_length * math.cos(positive_angle)
            positive_y -= positive_delta_y       # -=
            positive_x += positive_delta_x       # +=
            

        if Flag_negative == True:
            negative_angle = angle[IntRound(negative_y), IntRound(negative_x)]/180*math.pi
            negative_gray = graymap[IntRound(negative_y), IntRound(negative_x)]
            negative_delta_y = -step_length * math.sin(negative_angle)
            negative_delta_x = -step_length * math.cos(negative_angle)
            negative_y -= negative_delta_y       # -=
            negative_x += negative_delta_x       # +=
            

        # if out of canvas, stop
        if positive_y<0 or IntRound(positive_y)> H-1 or IntRound(positive_x)> W-1 or positive_x<0:
            Flag_positive = False
        if negative_y<0 or IntRound(negative_y)> H-1 or IntRound(negative_x)> W-1 or negative_x<0:
            Flag_negative = False



       
        

        if Flag_positive == True:
            grayscale_positive = graymap[IntRound(positive_y),IntRound(positive_x)]
            if len(path) < min_length: 
                path.append((positive_y, positive_x))
            elif grayscale_positive<=grayscale+threshold:
                path.append((positive_y, positive_x))
            else:
                Flag_positive = False     

        if Flag_negative == True:
            grayscale_negative = graymap[IntRound(negative_y),IntRound(negative_x)]
            if len(path) < min_length: 
                path.insert(0,(negative_y, negative_x))
            elif grayscale_negative<=grayscale+threshold:
                path.insert(0,(negative_y, negative_x))
            else:
                Flag_negative = False  
        
        if Flag_positive == False and Flag_negative == False:
            break

        if len(path)> max_length:
            break

    return path

# FindPath_V3() 根据ETF向量场搜索，根据起点的方向找直线
# graymap 灰度图, angle 向量场, y, x, step_length 步长, min_length 线的最短长度
# 灰度和起点的差的绝对值小于阈值
def FindPath_V3(graymap, angle, y, x, step_length = 1, min_length = 10, max_length=80, threshold = 25.5):

    (H,W) = graymap.shape       # image shape
    path = [[y,x]]              # path: a list to save points' trace [y,x]
    
    grayscale = graymap[IntRound(y),IntRound(x)]
    # grayscale = 0
    positive_angle = angle[IntRound(y),IntRound(x)]/180*math.pi
    negative_angle = angle[IntRound(y),IntRound(x)]/180*math.pi

    # direction positive and negative 
    positive_x, positive_y = x, y       # begin point for positive direction
    negative_x, negative_y = x, y       # begin point for negative direction
    Flag_positive = True                # stop condition
    Flag_negative = True                # stop condition
    while(1):
        if Flag_positive == True:
            # positive_angle = angle[IntRound(positive_y), IntRound(positive_x)]/180*math.pi
            positive_gray = graymap[IntRound(positive_y), IntRound(positive_x)]
            positive_delta_y = step_length * math.sin(positive_angle)
            positive_delta_x = step_length * math.cos(positive_angle)
            positive_y -= positive_delta_y       # -=
            positive_x += positive_delta_x       # +=
            

        if Flag_negative == True:
            # negative_angle = angle[IntRound(negative_y), IntRound(negative_x)]/180*math.pi
            negative_gray = graymap[IntRound(negative_y), IntRound(negative_x)]
            negative_delta_y = -step_length * math.sin(negative_angle)
            negative_delta_x = -step_length * math.cos(negative_angle)
            negative_y -= negative_delta_y       # -=
            negative_x += negative_delta_x       # +=
            

        # if out of canvas, stop
        if positive_y<0 or IntRound(positive_y)> H-1 or IntRound(positive_x)> W-1 or positive_x<0:
            Flag_positive = False
        if negative_y<0 or IntRound(negative_y)> H-1 or IntRound(negative_x)> W-1 or negative_x<0:
            Flag_negative = False



       
        

        if Flag_positive == True:
            grayscale_positive = graymap[IntRound(positive_y),IntRound(positive_x)]
            if len(path) < min_length: 
                path.append([positive_y, positive_x])
            elif grayscale_positive<=grayscale+threshold:
                path.append([positive_y, positive_x])
            else:
                Flag_positive = False     

        if Flag_negative == True:
            grayscale_negative = graymap[IntRound(negative_y),IntRound(negative_x)]
            if len(path) < min_length: 
                path.insert(0,[negative_y, negative_x])
            elif grayscale_negative<=grayscale+threshold:
                path.insert(0,[negative_y, negative_x])
            else:
                Flag_negative = False  
        
        if Flag_positive == False and Flag_negative == False:
            break

        if len(path)> max_length:
            break

    return path



# FindPath_V4() 根据起点的方向找直线
# 灰度和起点的差的绝对值小于阈值
# 方向和起点的差的绝对值小于阈值
def FindPath_V4(graymap, angle, y, x, step_length = 1, min_length = 10, max_length=80, threshold = 25.5, threshold_angle=20):

    (H,W) = graymap.shape       # image shape
    path = [[y,x]]              # path: a list to save points' trace [y,x]
    
    y = np.clip(y,0,H-1)
    x = np.clip(x,0,W-1)
    grayscale = graymap[IntRound(y),IntRound(x)]
    # grayscale = 0
    angle_origin = angle[IntRound(y),IntRound(x)]
    positive_angle = angle_origin/180*math.pi
    negative_angle = angle_origin/180*math.pi

    # direction positive and negative 
    positive_x, positive_y = x, y       # begin point for positive direction
    negative_x, negative_y = x, y       # begin point for negative direction
    Flag_positive = True                # stop condition
    Flag_negative = True                # stop condition
    while(1):
        if Flag_positive == True:
            # positive_angle = angle[IntRound(positive_y), IntRound(positive_x)]/180*math.pi
            positive_gray = graymap[IntRound(positive_y), IntRound(positive_x)]
            positive_delta_y = step_length * math.sin(positive_angle)
            positive_delta_x = step_length * math.cos(positive_angle)
            positive_y -= positive_delta_y       # -=
            positive_x += positive_delta_x       # +=
            

        if Flag_negative == True:
            # negative_angle = angle[IntRound(negative_y), IntRound(negative_x)]/180*math.pi
            negative_gray = graymap[IntRound(negative_y), IntRound(negative_x)]
            negative_delta_y = -step_length * math.sin(negative_angle)
            negative_delta_x = -step_length * math.cos(negative_angle)
            negative_y -= negative_delta_y       # -=
            negative_x += negative_delta_x       # +=
            

        # if out of canvas, stop
        if positive_y<0 or IntRound(positive_y)> H-1 or IntRound(positive_x)> W-1 or positive_x<0:
            Flag_positive = False
        if negative_y<0 or IntRound(negative_y)> H-1 or IntRound(negative_x)> W-1 or negative_x<0:
            Flag_negative = False



        if Flag_positive == True:
            grayscale_positive = graymap[IntRound(positive_y),IntRound(positive_x)]
            angle_positive = angle[IntRound(positive_y),IntRound(positive_x)]
            if len(path) < min_length: 
                path.append([positive_y, positive_x])
            elif grayscale_positive<=grayscale+threshold and (abs(angle_positive-angle_origin) < threshold_angle or (180-abs(angle_positive-angle_origin))<threshold_angle):
                    path.append([positive_y, positive_x])
            else:
                Flag_positive = False     

        if Flag_negative == True:
            grayscale_negative = graymap[IntRound(negative_y),IntRound(negative_x)]
            angle_negative = angle[IntRound(negative_y),IntRound(negative_x)]
            if len(path) < min_length: 
                path.insert(0,[negative_y, negative_x])
            elif grayscale_negative<=grayscale+threshold and (abs(angle_negative-angle_origin) < threshold_angle or (180-abs(angle_negative-angle_origin))<threshold_angle):
                    path.insert(0,[negative_y, negative_x])
            else:
                Flag_negative = False  
        
        if Flag_positive == False and Flag_negative == False:
            break

        if len(path)> max_length:
            break

    return path



# FindPath_V5() 根据起点的方向找直线
# 如果大于最小长度：灰度和起点的差的绝对值小于阈值
# 如果大于最大长度：方向和起点的差的绝对值小于阈值
def FindPath_V5(graymap, angle, y, x, step_length = 1, min_length = 10, max_length=80, threshold = 25.5, threshold_angle=20):

    (H,W) = graymap.shape       # image shape
    path = [[y,x]]              # path: a list to save points' trace [y,x]
    
    y = np.clip(y,0,H-1)
    x = np.clip(x,0,W-1)
    grayscale = graymap[IntRound(y),IntRound(x)]
    # grayscale = 0
    angle_origin = angle[IntRound(y),IntRound(x)]
    positive_angle = angle_origin/180*math.pi
    negative_angle = angle_origin/180*math.pi

    # direction positive and negative 
    positive_x, positive_y = x, y       # begin point for positive direction
    negative_x, negative_y = x, y       # begin point for negative direction
    Flag_positive = True                # stop condition
    Flag_negative = True                # stop condition
    while(1):
        if Flag_positive == True:
            # positive_angle = angle[IntRound(positive_y), IntRound(positive_x)]/180*math.pi
            positive_gray = graymap[IntRound(positive_y), IntRound(positive_x)]
            positive_delta_y = step_length * math.sin(positive_angle)
            positive_delta_x = step_length * math.cos(positive_angle)
            positive_y -= positive_delta_y       # -=
            positive_x += positive_delta_x       # +=
            

        if Flag_negative == True:
            # negative_angle = angle[IntRound(negative_y), IntRound(negative_x)]/180*math.pi
            negative_gray = graymap[IntRound(negative_y), IntRound(negative_x)]
            negative_delta_y = -step_length * math.sin(negative_angle)
            negative_delta_x = -step_length * math.cos(negative_angle)
            negative_y -= negative_delta_y       # -=
            negative_x += negative_delta_x       # +=
            

        # if out of canvas, stop
        if positive_y<0 or IntRound(positive_y)> H-1 or IntRound(positive_x)> W-1 or positive_x<0:
            Flag_positive = False
        if negative_y<0 or IntRound(negative_y)> H-1 or IntRound(negative_x)> W-1 or negative_x<0:
            Flag_negative = False



        if Flag_positive == True:
            grayscale_positive = graymap[IntRound(positive_y),IntRound(positive_x)]
            angle_positive = angle[IntRound(positive_y),IntRound(positive_x)]
            if len(path) < min_length: 
                path.append([positive_y, positive_x])
            elif grayscale_positive<=grayscale+threshold:
                if len(path)<max_length:
                    path.append([positive_y, positive_x])
                elif abs(angle_positive-angle_origin) < threshold_angle or (180-abs(angle_positive-angle_origin))<threshold_angle:
                    path.append([positive_y, positive_x])
                else:
                    Flag_positive = False  
            else:
                Flag_positive = False     

        if Flag_negative == True:
            grayscale_negative = graymap[IntRound(negative_y),IntRound(negative_x)]
            angle_negative = angle[IntRound(negative_y),IntRound(negative_x)]
            if len(path) < min_length: 
                path.insert(0,[negative_y, negative_x])
            elif grayscale_negative<=grayscale+threshold:
                if len(path)<max_length:
                    path.insert(0,[negative_y, negative_x])
                elif abs(angle_negative-angle_origin) < threshold_angle or (180-abs(angle_negative-angle_origin))<threshold_angle:
                    path.insert(0,[negative_y, negative_x])
                else:
                    Flag_negative = False 
            else:
                Flag_negative = False  
        
        if Flag_positive == False and Flag_negative == False:
            break

        if len(path)> max_length:
            break

    return path



# FindPath_V5Line() 根据patch起点的方向找直线
# 如果大于最小长度：灰度和起点的差的绝对值小于阈值
# 如果大于最大长度：方向和起点的差的绝对值小于阈值
def FindPath_V5Line(graymap, angle_origin, angle, y, x, step_length = 1, min_length = 10, max_length=80, threshold = 25.5, threshold_angle=20):

    (H,W) = graymap.shape       # image shape
    path = [[y,x]]              # path: a list to save points' trace [y,x]
    
    y = np.clip(y,0,H-1)
    x = np.clip(x,0,W-1)
    grayscale = graymap[IntRound(y),IntRound(x)]
    # grayscale = 0
    # angle_origin = angle[IntRound(y),IntRound(x)]
    positive_angle = angle_origin/180*math.pi
    negative_angle = angle_origin/180*math.pi

    # direction positive and negative 
    positive_x, positive_y = x, y       # begin point for positive direction
    negative_x, negative_y = x, y       # begin point for negative direction
    Flag_positive = True                # stop condition
    Flag_negative = True                # stop condition
    while(1):
        if Flag_positive == True:
            # positive_angle = angle[IntRound(positive_y), IntRound(positive_x)]/180*math.pi
            positive_gray = graymap[IntRound(positive_y), IntRound(positive_x)]
            positive_delta_y = step_length * math.sin(positive_angle)
            positive_delta_x = step_length * math.cos(positive_angle)
            positive_y -= positive_delta_y       # -=
            positive_x += positive_delta_x       # +=
            

        if Flag_negative == True:
            # negative_angle = angle[IntRound(negative_y), IntRound(negative_x)]/180*math.pi
            negative_gray = graymap[IntRound(negative_y), IntRound(negative_x)]
            negative_delta_y = -step_length * math.sin(negative_angle)
            negative_delta_x = -step_length * math.cos(negative_angle)
            negative_y -= negative_delta_y       # -=
            negative_x += negative_delta_x       # +=
            

        # if out of canvas, stop
        if positive_y<0 or IntRound(positive_y)> H-1 or IntRound(positive_x)> W-1 or positive_x<0:
            Flag_positive = False
        if negative_y<0 or IntRound(negative_y)> H-1 or IntRound(negative_x)> W-1 or negative_x<0:
            Flag_negative = False



        if Flag_positive == True:
            grayscale_positive = graymap[IntRound(positive_y),IntRound(positive_x)]
            angle_positive = angle[IntRound(positive_y),IntRound(positive_x)]
            if len(path) < min_length: 
                path.append([positive_y, positive_x])
            elif grayscale_positive<=grayscale+threshold:
                if len(path)<max_length:
                    path.append([positive_y, positive_x])
                elif abs(angle_positive-angle_origin) < threshold_angle or (180-abs(angle_positive-angle_origin))<threshold_angle:
                    path.append([positive_y, positive_x])
                else:
                    Flag_positive = False  
            else:
                Flag_positive = False     

        if Flag_negative == True:
            grayscale_negative = graymap[IntRound(negative_y),IntRound(negative_x)]
            angle_negative = angle[IntRound(negative_y),IntRound(negative_x)]
            if len(path) < min_length: 
                path.insert(0,[negative_y, negative_x])
            elif grayscale_negative<=grayscale+threshold:
                if len(path)<max_length:
                    path.insert(0,[negative_y, negative_x])
                elif abs(angle_negative-angle_origin) < threshold_angle or (180-abs(angle_negative-angle_origin))<threshold_angle:
                    path.insert(0,[negative_y, negative_x])
                else:
                    Flag_negative = False 
            else:
                Flag_negative = False  
        
        if Flag_positive == False and Flag_negative == False:
            break

        if len(path)> max_length:
            break

    return path
    


def FindArea_Rec(graymap, hsvmap, angle, y, x, step_length = 1, min_length = 10, max_length=80, threshold = 25.5, threshold_hsv= [10,10,10], threshold_angle=20):
    def color_condition(HSV, hsv, threshold_hsv):
        # if abs(HSV[0]-hsv[0]) <= threshold_hsv[0] and abs(HSV[1]-hsv[1]) <= threshold_hsv[1] and abs(HSV[2]-hsv[2]) <= threshold_hsv[2]:
        # if abs(HSV[1]-hsv[1]) <= threshold_hsv[1] and abs(HSV[2]-hsv[2]) <= threshold_hsv[2]:
        if (abs(float(HSV[0])-float(hsv[0])) <= threshold_hsv[0] or abs(float(HSV[0])+180-float(hsv[0])) <= threshold_hsv[0] or abs(float(HSV[0])-180-float(hsv[0])) <= threshold_hsv[0]) \
            and float(HSV[2]) >= float(hsv[2])-threshold_hsv[2]:
        # if float(HSV[2]) >= float(hsv[2])-threshold_hsv[2]:
        # if abs(HSV[0]-hsv[0]) <= threshold_hsv[0] and hsv[2] > HSV[2]:
            return True
        else:
            return False

    (h0,w0) = graymap.shape       # image shape
    grayscale_0 = graymap[y,x]
    (h,s,v) = hsvmap[y,x]
    angle_0 = angle[y,x]

    y_begin = max(0, y-max_length) 
    y_end =  min(h0, y+max_length)
    x_begin = max(0, x-max_length) 
    x_end =  min(w0, x+max_length)
    
    patch_hsv = hsvmap[y_begin:y_end, x_begin:x_end].astype("float32")
    mask1 = abs(patch_hsv[:,:,0] - h) <= threshold_hsv[0]
    mask2 = abs(patch_hsv[:,:,0]+180 - h) <= threshold_hsv[0]
    mask3 = abs(patch_hsv[:,:,0]-180 - h) <= threshold_hsv[0]
    mask4 = patch_hsv[:,:,2] >= v-threshold_hsv[2]

    patch_angle = angle[y_begin:y_end, x_begin:x_end]
    mask5 = abs(patch_angle-angle_0) <= threshold_angle 
    mask6 = (180-abs(patch_angle-angle_0)) <= threshold_angle
  
    mask = (mask1+mask2+mask3)*mask4*(mask5+mask6) 
    mask =  mask.astype("uint8")*255
    # cv2.imwrite(output_path + "/mask.png", mask) 
    cv2.imshow('mask', mask)
    cv2.waitKey(0)

    return mask

# FindPath_V6() 根据起点找矩形，长度方向为ETF方向，宽度方向为正交方向
# 如果大于最小长度：灰度和起点的差的绝对值小于阈值
# 如果大于最大长度：方向和起点的差的绝对值小于阈值
def FindPath_V6(graymap, hsvmap, angle, y, x, step_length = 1, min_length = 10, max_length=80, threshold = 25.5, threshold_hsv= [10,10,10], threshold_angle=20):

    def color_condition(HSV, hsv, threshold_hsv):
        # if abs(HSV[0]-hsv[0]) <= threshold_hsv[0] and abs(HSV[1]-hsv[1]) <= threshold_hsv[1] and abs(HSV[2]-hsv[2]) <= threshold_hsv[2]:
        # if abs(HSV[1]-hsv[1]) <= threshold_hsv[1] and abs(HSV[2]-hsv[2]) <= threshold_hsv[2]:
        if (abs(float(HSV[0])-float(hsv[0])) <= threshold_hsv[0] or abs(float(HSV[0])+180-float(hsv[0])) <= threshold_hsv[0] or abs(float(HSV[0])-180-float(hsv[0])) <= threshold_hsv[0]) \
            and float(HSV[2]) >= float(hsv[2])-threshold_hsv[2]:
        # if float(HSV[2]) >= float(hsv[2])-threshold_hsv[2]:
        # if abs(HSV[0]-hsv[0]) <= threshold_hsv[0] and hsv[2] > HSV[2]:
            return True
        else:
            return False

    (H,W) = graymap.shape       # image shape
    path = [[y,x]]              # path: a list to save points' trace [y,x]
    
    y = np.clip(y,0,H-1)
    x = np.clip(x,0,W-1)
    grayscale = graymap[IntRound(y),IntRound(x)]
    hsv_0 = hsvmap[IntRound(y),IntRound(x)]
    # grayscale = 0
    angle_origin = angle[IntRound(y),IntRound(x)]
    positive_angle = angle_origin/180*math.pi
    negative_angle = angle_origin/180*math.pi

    # direction positive and negative 
    positive_x, positive_y = x, y       # begin point for positive direction
    negative_x, negative_y = x, y       # begin point for negative direction
    Flag_positive = True                # stop condition
    Flag_negative = True                # stop condition
    while(1):
        if Flag_positive == True:
            # positive_angle = angle[IntRound(positive_y), IntRound(positive_x)]/180*math.pi
            positive_gray = graymap[IntRound(positive_y), IntRound(positive_x)]
            positive_delta_y = step_length * math.sin(positive_angle)
            positive_delta_x = step_length * math.cos(positive_angle)
            positive_y -= positive_delta_y       # -=
            positive_x += positive_delta_x       # +=
            

        if Flag_negative == True:
            # negative_angle = angle[IntRound(negative_y), IntRound(negative_x)]/180*math.pi
            negative_gray = graymap[IntRound(negative_y), IntRound(negative_x)]
            negative_delta_y = -step_length * math.sin(negative_angle)
            negative_delta_x = -step_length * math.cos(negative_angle)
            negative_y -= negative_delta_y       # -=
            negative_x += negative_delta_x       # +=
            

        # if out of canvas, stop
        if positive_y<0 or IntRound(positive_y)> H-1 or IntRound(positive_x)> W-1 or positive_x<0:
            Flag_positive = False
        if negative_y<0 or IntRound(negative_y)> H-1 or IntRound(negative_x)> W-1 or negative_x<0:
            Flag_negative = False



        if Flag_positive == True:
            grayscale_positive = graymap[IntRound(positive_y),IntRound(positive_x)]
            hsv_positive  = hsvmap[IntRound(positive_y),IntRound(positive_x)]
            angle_positive = angle[IntRound(positive_y),IntRound(positive_x)]
            if len(path) < min_length: 
                path.append([positive_y, positive_x])
            # elif color_condition(hsv_0,hsv_positive,threshold_hsv)==True:
            elif color_condition(hsv_0,hsv_positive,threshold_hsv)==True:
                if len(path)<max_length:
                    path.append([positive_y, positive_x])
                elif abs(angle_positive-angle_origin) < threshold_angle or (180-abs(angle_positive-angle_origin))<threshold_angle:
                    path.append([positive_y, positive_x])
                else:
                    Flag_positive = False  
            else:
                Flag_positive = False     

        if Flag_negative == True:
            grayscale_negative = graymap[IntRound(negative_y),IntRound(negative_x)]
            hsv_negative  = hsvmap[IntRound(negative_y),IntRound(negative_x)]
            angle_negative = angle[IntRound(negative_y),IntRound(negative_x)]
            if len(path) < min_length: 
                path.insert(0,[negative_y, negative_x])
            # elif color_condition(hsv_0,hsv_negative,threshold_hsv)==True:
            elif color_condition(hsv_0,hsv_negative,threshold_hsv)==True:
                if len(path)<max_length:
                    path.insert(0,[negative_y, negative_x])
                elif abs(angle_negative-angle_origin) < threshold_angle or (180-abs(angle_negative-angle_origin))<threshold_angle:
                    path.insert(0,[negative_y, negative_x])
                else:
                    Flag_negative = False 
            else:
                Flag_negative = False  
        
        if Flag_positive == False and Flag_negative == False:
            break

        if len(path)> max_length:
            break

    return path


def FindPath_V7(graymap, hsvmap, angle, y, x, step_length = 1, min_length = 10, max_length=80, threshold_hsv= [10,10,10]):

    def color_condition(HSV, hsv, threshold_hsv):
        # if (abs(float(HSV[0])-float(hsv[0])) <= threshold_hsv[0] or abs(float(HSV[0])+180-float(hsv[0])) <= threshold_hsv[0] or abs(float(HSV[0])-180-float(hsv[0])) <= threshold_hsv[0]) \
        #     and float(HSV[2]) >= float(hsv[2])-threshold_hsv[2]:
        if (abs(float(HSV[0])-float(hsv[0])) <= threshold_hsv[0] or abs(float(HSV[0])+180-float(hsv[0])) <= threshold_hsv[0] or abs(float(HSV[0])-180-float(hsv[0])) <= threshold_hsv[0]) \
            and abs(float(HSV[2]) - float(hsv[2])) <= threshold_hsv[2]:
            return True
        else:
            return False

    (H,W) = graymap.shape       # image shape
    path = [[y,x]]              # path: a list to save points' trace [y,x]
    length_positive = 0
    length_negative = 0 

    hsv_0 = hsvmap[IntRound(y),IntRound(x)]
    angle_origin = angle[IntRound(y),IntRound(x)]
    angle_origin = angle_origin/180*math.pi

    # direction positive and negative 
    positive_x, positive_y = x, y       # begin point for positive direction
    negative_x, negative_y = x, y       # begin point for negative direction
    Flag_positive = True                # stop condition
    Flag_negative = True                # stop condition
    while(1):
        delta_y = step_length * math.sin(angle_origin)
        delta_x = step_length * math.cos(angle_origin)       
        if Flag_positive == True:
            positive_y -= delta_y       # -=
            positive_x += delta_x       # +=
        if Flag_negative == True:
            negative_y += delta_y       # -=
            negative_x -= delta_x       # +=
            

        # if out of canvas, stop
        if positive_y<0 or IntRound(positive_y)> H-1 or IntRound(positive_x)> W-1 or positive_x<0:
            Flag_positive = False
        if negative_y<0 or IntRound(negative_y)> H-1 or IntRound(negative_x)> W-1 or negative_x<0:
            Flag_negative = False



        if Flag_positive == True:
            hsv_positive  = hsvmap[IntRound(positive_y),IntRound(positive_x)]
            if length_positive < min_length: 
                path.append([positive_y, positive_x])
                length_positive += 1
            elif color_condition(hsv_0,hsv_positive,threshold_hsv)==True:
                if length_positive < max_length:
                    path.append([positive_y, positive_x])
                    length_positive += 1
                else:
                    Flag_positive = False  
            else:
                Flag_positive = False     

        if Flag_negative == True:
            hsv_negative  = hsvmap[IntRound(negative_y),IntRound(negative_x)]
            if length_negative < min_length: 
                path.insert(0,[negative_y, negative_x])
                length_negative += 1
            elif color_condition(hsv_0,hsv_negative,threshold_hsv)==True:
                if length_negative < max_length:
                    path.insert(0,[negative_y, negative_x])
                    length_negative += 1
                else:
                    Flag_negative = False 
            else:
                Flag_negative = False  
        
        if Flag_positive == False and Flag_negative == False:
            break


    return path, length_negative, length_positive





# FindPath_V6() 根据起点找矩形，长度方向为ETF方向，宽度方向为正交方向
# 如果大于最小长度：灰度和起点的差的绝对值小于阈值
# 如果大于最大长度：方向和起点的差的绝对值小于阈值
def FindPath_V6Line(graymap, hsvmap, angle_origin, angle, y, x, step_length = 1, min_length = 10, max_length=80, threshold = 25.5, threshold_hsv= [10,10,10], threshold_angle=20):

    def color_condition(HSV, hsv, threshold_hsv):
        # if abs(HSV[0]-hsv[0]) <= threshold_hsv[0] and abs(HSV[1]-hsv[1]) <= threshold_hsv[1] and abs(HSV[2]-hsv[2]) <= threshold_hsv[2]:
        # if abs(HSV[1]-hsv[1]) <= threshold_hsv[1] and abs(HSV[2]-hsv[2]) <= threshold_hsv[2]:
        if (abs(float(HSV[0])-float(hsv[0])) <= threshold_hsv[0] or abs(float(HSV[0])+180-float(hsv[0])) <= threshold_hsv[0] or abs(float(HSV[0])-180-float(hsv[0])) <= threshold_hsv[0]) \
            and float(HSV[2]) >= float(hsv[2])-threshold_hsv[2]:
        # if float(HSV[2]) >= float(hsv[2])-threshold_hsv[2]:
        # if abs(HSV[0]-hsv[0]) <= threshold_hsv[0] and hsv[2] > HSV[2]:
            return True
        else:
            return False

    (H,W) = graymap.shape       # image shape
    path = [[y,x]]              # path: a list to save points' trace [y,x]
    
    y = np.clip(y,0,H-1)
    x = np.clip(x,0,W-1)
    grayscale = graymap[IntRound(y),IntRound(x)]
    hsv_0 = hsvmap[IntRound(y),IntRound(x)]
    # grayscale = 0
    # angle_origin = angle[IntRound(y),IntRound(x)]
    positive_angle = angle_origin/180*math.pi
    negative_angle = angle_origin/180*math.pi

    # direction positive and negative 
    positive_x, positive_y = x, y       # begin point for positive direction
    negative_x, negative_y = x, y       # begin point for negative direction
    Flag_positive = True                # stop condition
    Flag_negative = True                # stop condition
    while(1):
        if Flag_positive == True:
            # positive_angle = angle[IntRound(positive_y), IntRound(positive_x)]/180*math.pi
            positive_gray = graymap[IntRound(positive_y), IntRound(positive_x)]
            positive_delta_y = step_length * math.sin(positive_angle)
            positive_delta_x = step_length * math.cos(positive_angle)
            positive_y -= positive_delta_y       # -=
            positive_x += positive_delta_x       # +=
            

        if Flag_negative == True:
            # negative_angle = angle[IntRound(negative_y), IntRound(negative_x)]/180*math.pi
            negative_gray = graymap[IntRound(negative_y), IntRound(negative_x)]
            negative_delta_y = -step_length * math.sin(negative_angle)
            negative_delta_x = -step_length * math.cos(negative_angle)
            negative_y -= negative_delta_y       # -=
            negative_x += negative_delta_x       # +=
            

        # if out of canvas, stop
        if IntRound(positive_y)<0 or IntRound(positive_y)> H-1 or IntRound(positive_x)> W-1 or IntRound(positive_x)<0:
            Flag_positive = False
        if IntRound(negative_y)<0 or IntRound(negative_y)> H-1 or IntRound(negative_x)> W-1 or IntRound(negative_x)<0:
            Flag_negative = False



        if Flag_positive == True:
            grayscale_positive = graymap[IntRound(positive_y),IntRound(positive_x)]
            hsv_positive  = hsvmap[IntRound(positive_y),IntRound(positive_x)]
            angle_positive = angle[IntRound(positive_y),IntRound(positive_x)]
            if len(path) < min_length: 
                path.append([positive_y, positive_x])
            # elif color_condition(hsv_0,hsv_positive,threshold_hsv)==True:
            elif color_condition(hsv_0,hsv_positive,threshold_hsv)==True:
                if len(path)<max_length:
                    path.append([positive_y, positive_x])
                elif abs(angle_positive-angle_origin) < threshold_angle or (180-abs(angle_positive-angle_origin))<threshold_angle:
                    path.append([positive_y, positive_x])
                else:
                    Flag_positive = False  
            else:
                Flag_positive = False     

        if Flag_negative == True:
            grayscale_negative = graymap[IntRound(negative_y),IntRound(negative_x)]
            hsv_negative  = hsvmap[IntRound(negative_y),IntRound(negative_x)]
            angle_negative = angle[IntRound(negative_y),IntRound(negative_x)]
            if len(path) < min_length: 
                path.insert(0,[negative_y, negative_x])
            # elif color_condition(hsv_0,hsv_negative,threshold_hsv)==True:
            elif color_condition(hsv_0,hsv_negative,threshold_hsv)==True:
                if len(path)<max_length:
                    path.insert(0,[negative_y, negative_x])
                elif abs(angle_negative-angle_origin) < threshold_angle or (180-abs(angle_negative-angle_origin))<threshold_angle:
                    path.insert(0,[negative_y, negative_x])
                else:
                    Flag_negative = False 
            else:
                Flag_negative = False  
        
        if Flag_positive == False and Flag_negative == False:
            break

        if len(path)> max_length:
            break

    return path

