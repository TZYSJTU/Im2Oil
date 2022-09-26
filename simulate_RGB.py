import cv2
import numpy as np

from simulate import *

constant_length = 1000

def GetBrush_HSV_Alpha(width, length, mean, brush):
    (h,w) = brush.shape
    canvas = Gassian_HSV(size=(h,w,3), mean=mean, var = [0,0,0])  # hsv 颜色空间测试
    
    # canvas_V = canvas[:,:,2].astype("float32") * brush.astype("float32")/255
    # canvas[:,:,2] = np.uint8(canvas_V) 

    # canvas[brush<120,2] = 254
    # canvas[brush<120,1] = 0

    canvas = cv2.resize(canvas,(length, width))
    alpha = cv2.resize(brush,(length, width))
    # canvas = cv2.cvtColor(canvas, cv2.COLOR_HSV2BGR)
    # cv2.imshow('brush', canvas)
    # cv2.waitKey(0)
    return canvas, alpha

def GetBrush_HSV(width, length, mean, brush):
    (h,w) = brush.shape
    canvas = Gassian_HSV(size=(h,w,3), mean=mean, var = [0,0,0])  # hsv 颜色空间测试
    canvas_V = canvas[:,:,2].astype("float32") * (brush.astype("float32")/255)
    
    # min_V = brush[brush>0].min()
    mean_V = brush[brush>0].mean()
    rate_1 = mean_V/255
    rate_2 = canvas_V.max()/255
    rate = max(rate_1, rate_2)
    canvas_V /=  rate
    # canvas_V = np.clip(canvas_V, 0, 255)
    mean_ = canvas_V[brush>0].mean()

    canvas[:,:,2] = np.uint8(canvas_V) 
    canvas = cv2.resize(canvas,(length, width))

    mask = np.ones(brush.shape) 
    mask[brush==0] = 0
    mask = cv2.resize(mask,(length, width))
    # mask[mask<1] = 0
    # canvas = cv2.cvtColor(canvas, cv2.COLOR_HSV2BGR)
    # cv2.imshow('mask', np.uint8(mask*255))
    # cv2.waitKey(0)
    return canvas, mask

# def GetBrush_HSV(width, length, mean, brush):
#     (h,w) = brush.shape
#     canvas = Gassian_HSV(size=(h,w,3), mean=mean, var = [0,0,0])  # hsv 颜色空间测试
#     canvas_V = canvas[:,:,2].astype("float32") * brush.astype("float32")/255
#     canvas[:,:,2] = np.uint8(canvas_V) 

#     canvas[brush<120,2] = 254
#     canvas[brush<120,1] = 0

#     canvas = cv2.resize(canvas,(length, width))

#     mask = np.ones(brush.shape) 
#     mask[brush==0] = 0
#     mask = cv2.resize(mask,(length, width))
#     mask[mask<1] = 0
#     # canvas = cv2.cvtColor(canvas, cv2.COLOR_HSV2BGR)
#     # cv2.imshow('mask', np.uint8(mask*255))
#     # cv2.waitKey(0)
#     return canvas, mask

def GetParallel_HSV_new(width, length, period, grayscale):
    distribution = ChooseDistribution(period=period, Grayscale=grayscale)
    patch = GetParallel(distribution=distribution, height=width, length=length, period=period)
    patch = cv2.cvtColor(patch,cv2.COLOR_GRAY2BGR)
    patch = cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
    return patch

def GetParallel_HSV(width, length, period, grayscale):
    distribution = ChooseDistribution(period=period, Grayscale=grayscale)
    patch = GetParallel(distribution=distribution, height=width, length=length, period=period)
    patch = cv2.cvtColor(patch,cv2.COLOR_GRAY2BGR)
    patch = cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)

    mask = np.ones((width, length)) 
    return patch, mask


# H范围是[0,179](若在180-255，等于减去180)，S范围是[0,255]，V范围是[0,255]
def Gassian_HSV(size, mean=[0,0,255], var=[0,0,0]):
    # size is [h,w,3]
    # mean is [h,s,v]
    # var is  [h,s,v]
    norm = np.random.randn(*size)
    denorm = norm * np.sqrt(var) + mean
    return np.uint8(np.round(np.clip(denorm,0,255)))





def Getline_HSV(period, length, mean):
    patch = np.zeros((period,length,3)) + mean
    mask = Getline_Mask_oil(period,length,var=1)

    patch[:,:,1:] *= mask
    patch[:,:,2:] += (1-mask)*255

    return np.uint8(np.round(np.clip(patch,0,255)))



def Getline_visialization(period, length, mean, var=[0,0,0]):
    # if length < constant_length:  # if length is shorter than constant_length, lines are Aligned
    # patch = Gassian_HSV((2*period, length,3)) 
    # begin = 0
    # end = 1
    # for i in range(period):
    patch=Gassian((period,length,3), mean, var)

    # print(patch[0])
    # patch[:,:,0] = np.clip(patch[:,:,0],0,180)
    # patch[:period] = Attenuation(patch[:period], period=period,begin=begin, end=end)
    # patch = Distortion(patch, begin=begin, end=end)

    return np.uint8(np.round(np.clip(patch,0,255)))


def GetStroke_HSV_pencil(width, length, mean, stroke):
    (h,w) = stroke.shape
    stroke = cv2.resize(stroke,(length,width))
    # ret2,th2 = cv2.threshold(stroke,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # print(stroke[stroke<ret2].mean())    # 186


    canvas = Gassian_HSV(size=(width,length,3), mean=mean, var = [0,0,0])  # hsv 颜色空间测试
    canvas_V = canvas[:,:,2].astype("float32") * stroke.astype("float32")/255
    canvas[:,:,2] = np.uint8(canvas_V) 

    canvas[stroke>218,2] = 255
    canvas[stroke>218,1] = 0
    # canvas = cv2.cvtColor(canvas, cv2.COLOR_HSV2BGR)
    # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    return canvas


def Getline_HSV_pencil(period, length, mean, var=[0,0,0]):
    # if length < constant_length:  # if length is shorter than constant_length, lines are Aligned
    # patch = Gassian_HSV((2*period, length,3)) 
    # begin = 0
    # end = 1
    # for i in range(period):
    patch=Gassian((period,length,3), mean, var)
    mask = Getline_Mask_pencil(period,length,var=1)
    patch[:,:,1:] *= mask

    patch[:,:,2:] += (1-mask)*255
        # print(patch[0])
    # patch[:,:,0] = np.clip(patch[:,:,0],0,180)
    # patch[:period] = Attenuation(patch[:period], period=period,begin=begin, end=end)
    # patch = Distortion(patch, begin=begin, end=end)

    return np.uint8(np.round(np.clip(patch,0,255)))


# pencil
def Getline_Mask_pencil(period, length, var=1):
    mask = np.zeros((period, length, 1))
    if period % 2 == 1: # 宽度为奇数
        for i in range (period//2+1):
            mask[i] = Gassian_HSV(size=(1, length, 1), mean=period/2-(period/2-i)*2, var = period/2) 
            mask[-i-1] = Gassian_HSV(size=(1, length, 1), mean=period/2-(period/2-i)*2, var = period/2) 
    else: # 宽度为偶数
        for i in range (period//2):
            mask[i] = Gassian_HSV(size=(1, length, 1), mean=period/2-(period/2-i)*2, var = period/2) 
            mask[-i-1] = Gassian_HSV(size=(1, length, 1), mean=period/2-(period/2-i)*2, var = period/2) 
    # 两侧
    for i in range (2*period):
       #  p = (i+1) / (2*period)
        # numpy.random.randint(low = -i, high=2*period, size=(period, 1, 1))
        mask[:,i,None]  *= np.random.randint(low = i+2-2*period, high=i+1+1, size=(period, 1, 1))
        mask[:,-i-1,None] *= np.random.randint(low = i+2-2*period, high=i+1+1, size=(period, 1, 1))
    
    # # 圆形
    # for i in range (2*period):
    #     R = 17/4*period
    #     Y = R - np.sqrt(R**2 - (2*period-i-1)**2)
    #     Y = int(Y)+1
    #     mask[:Y,i] = 0
    #     mask[-Y:,i] = 0
    #     mask[:Y,-i-1] = 0
    #     mask[-Y:,-i-1] = 0

    mask[mask>0] = 1 # 概率等于 积分正半轴
    mask[mask<0] = 0    

    return np.uint8(mask)

def Gassian_Mask(size, mean=0, var=1):
    # size is [h,w,3]
    # mean is [h,s,v]
    # var is  [h,s,v]
    norm = np.random.randn(*size)
    denorm = norm * np.sqrt(var) + mean
    return denorm

# oil
def Getline_Mask_oil(period, length, var=1):
    mask = np.zeros((period, length, 1))
    if period % 2 == 1: # 宽度为奇数
        for i in range (period//2+1):
            mask[i] = Gassian_Mask(size=(1, length, 1), mean=(period/2-(period/2-i)*2), var = period) 
            mask[-i-1] = Gassian_Mask(size=(1, length, 1), mean=(period/2-(period/2-i)*2), var = period) 
    else: # 宽度为偶数
        for i in range (period//2):
            # mask[i] = np.random.uniform(low = i+1-0.5*period, high=i+1, size=(1, length, 1))
            # mask[-i-1] = np.random.uniform(low = i+1-0.5*period, high=i+1, size=(1, length, 1))

            mask[i] = Gassian_Mask(size=(1, length, 1), mean=period/2-(period/2-i)*2, var = period) 
            mask[-i-1] = Gassian_Mask(size=(1, length, 1), mean=period/2-(period/2-i)*2, var = period) 
    mask[mask>0] = 1 # 概率等于 积分正半轴
    mask[mask<0] = 0  
    # 两侧
    for i in range (1*period):
       #  p = (i+1) / (2*period)
        # numpy.random.randint(low = -i, high=2*period, size=(period, 1, 1))
        mask[:,i,None]  *= np.random.uniform(low = i+1-1*period, high=i+1, size=(period, 1, 1))
        mask[:,-i-1,None] *= np.random.uniform(low = i+1-1*period, high=i+1, size=(period, 1, 1))
    
    # # 圆形
    # for i in range (2*period):
    #     R = 17/4*period
    #     Y = R - np.sqrt(R**2 - (2*period-i-1)**2)
    #     Y = int(Y)+1
    #     mask[:Y,i] = 0
    #     mask[-Y:,i] = 0
    #     mask[:Y,-i-1] = 0
    #     mask[-Y:,-i-1] = 0

    mask[mask>0] = 1 # 概率等于 积分正半轴
    mask[mask<0] = 0    

    return np.uint8(mask)






def GetPatch_HSV(num, width, length, SSAA, hsv):
    patch = np.zeros((width+(num-1)*SSAA, length,3)) + [int(hsv[0]),0,255]
    for i in range(num):
        line = Getline_HSV(period=width, length=length,mean=hsv)
        temp = patch[i*SSAA:i*SSAA+width,:,:]
        m = (line[:,:,2:]>temp[:,:,2:])
        patch[i*SSAA:i*SSAA+width,:,:] = m*temp + (1-m)*line   

    # cv2.imshow('patch', cv2.cvtColor(patch, cv2.COLOR_HSV2BGR))
    # cv2.waitKey(0)

    return np.uint8(np.round(np.clip(patch,0,255)))







def Gassian(size, mean = 0, var = 0):
    # size is [h,w]
    # mean and var is a number
    norm = np.random.randn(*size)
    denorm = norm * np.sqrt(var) + mean
    return np.uint8(np.round(np.clip(denorm,0,255)))

def Getline(distribution, length):
    period = distribution.shape[0]
    if length < constant_length:  # if length is too short, lines are Aligned
        patch = Gassian((2*period, length), mean=255, var = 3) 
        begin = 0
        end = 1
        for i in range(period):
            patch[i]=Gassian((1,length), mean=distribution[i,0], var=distribution[i,1])

    else:           # if length is't too short, lines is't Aligned
        patch = Gassian((2*period, length+4*period), mean=255, var = 3) 

        begin = Gassian((1,1), mean=2.0*period, var=2*period)
        # egin = Gassian((1,1), mean=2.0*period, var=0)
        begin = np.uint8(np.round(np.clip(begin,0,4*period)))
        begin = int(begin[0,0])
        end = Gassian((1,1), mean=2.0*period, var=2*period)
        # end = Gassian((1,1), mean=2.0*period, var=0)
        end = np.uint8(np.round(np.clip(end,1,4*period+1)))
        end = int(end[0,0])

        real_length = length+4*period-end-begin
        for i in range(period):
            patch[i,begin:-end]=Gassian((1,real_length), mean=distribution[i,0], var=distribution[i,1])

    patch = Attenuation(patch, period=period,begin=begin, end=end)
    patch = Distortion(patch, begin=begin, end=end)

    return np.uint8(np.round(np.clip(patch,0,255)))

def Attenuation(patch, period, begin, end):
    order = int((patch.shape[1]-begin-end)/2)+1
    radius = (period-1)/2
    canvas = Gassian((patch.shape[0], patch.shape[1]), mean=250, var=3)
    patch = np.float32(patch)
    canvas = np.float32(canvas)
    for i in range(begin, patch.shape[1]-end+1):
        for j in range(period):
            a = np.abs((1.0-(i-begin)/order)**2)/3
            b = np.abs((1.0-j/radius)**2)*1
            patch[j,i] += (canvas[j,i]-patch[j,i])*np.sqrt(a+b)/1.5
            # patch[j,i] +=  0.75*(canvas[j,i]-patch[j,i]) * (np.abs((1.0-(i-begin)/order)**2))**0.5

    return np.uint8(np.round(np.clip(patch,0,255)))
# def Attenuation(patch, period, begin, end):
#     patch = np.float32(patch)
#     if period % 2 == 1: # 宽度为奇数
#         for i in range (period//2):
#             patch[i,:,1] *= 1-(period//2-i)/(period/2)
#             patch[-i,:,1] *= 1-(period//2-i)/(period/2)
#         patch[period//2,:,1] *= 1-(period//2-i)/(period/2)
#     else: # 宽度为偶数
#         for i in range (period//2):
#             patch[i,:,1] *= 1-(period//2-i)/period
#             patch[-i,:,1] *= 1-(period//2-i)/period

#     return np.uint8(np.round(np.clip(patch,0,255)))


# def Distortion(patch, begin, end):
#     # begin 正数第几个像素 end 倒数第几个像素
#     height = int(patch.shape[0]/2)
#     length = patch.shape[1]
#     patch = np.float32(patch)
#     patch_copy = patch.copy()

#     # central = ((length-begin-end)/2+begin) + np.random.randn()*length/30
#     central = ((length-begin-end)/2+begin) # 最中间的像素
#     if length>100:
#         radius = length**2/(4*height)
#     else:
#         radius = length**2/(2*height)
        
#     for i in range(length):
#         offset = ((central-i)**2)/(2*radius) 
#         int_offset = int(offset)
#         decimal_offset = offset-int_offset
#         for j in range(height):
#             if j>int_offset:
#                 patch[j,i]=(decimal_offset*patch_copy[j-1-int_offset,i]+(1-decimal_offset)*patch_copy[j-int_offset,i])
#             else:
#                 patch[j,i,2]= np.random.randn() * np.sqrt(3) + 255
   
#     patch_copy = patch.copy()
#     if length>100:
#         for i in range(length):
#             offset = ((central-i)**2)/(2*radius) 
#             int_offset = int(offset)
#             decimal_offset = offset-int_offset
#             for j in range(patch.shape[0]):
#                 if j>int_offset:
#                     patch[j,i]=(decimal_offset*patch_copy[j-1-int_offset,i]+(1-decimal_offset)*patch_copy[j-int_offset,i])
#                 else:
#                     patch[j,i,2]= np.random.randn() * np.sqrt(3) + 255
#         # else:
#         #     radius = length**2/(4*height)
#         #     for i in range(length):
#         #         offset = ((central-i)**2)/(2*radius) 
#         #         int_offset = int(offset)
#         #         decimal_offset = offset-int_offset
#         #         for j in range(patch.shape[0]):
#         #             if j>int_offset:
#         #                 patch[j,i]=int(decimal_offset*patch_copy[j-1-int_offset,i]+(1-decimal_offset)*patch_copy[j-int_offset,i])
#         #             else:
#         #                 patch[j,i]= np.random.randn() * np.sqrt(3) + 255            


#     return np.uint8(np.round(np.clip(patch,0,255)))
def Distortion(patch,begin,end):
    height = int(patch.shape[0]/2)
    length = patch.shape[1]
    patch = np.float32(patch)
    patch_copy = patch.copy()

    # central = ((length-begin-end)/2+begin) + np.random.randn()*length/30
    central = ((length-begin-end)/2+begin)
    if length>100:
        radius = length**2/(4*height)
    else:
        radius = length**2/(2*height)
        
    for i in range(length):
        offset = ((central-i)**2)/(2*radius) 
        int_offset = int(offset)
        decimal_offset = offset-int_offset
        for j in range(height):
            if j>int_offset:
                patch[j,i]=int(decimal_offset*patch_copy[j-1-int_offset,i]+(1-decimal_offset)*patch_copy[j-int_offset,i])
            else:
                patch[j,i]= np.random.randn() * np.sqrt(3) + 250
   
    patch_copy = patch.copy()
    if length>100:
        for i in range(length):
            offset = ((central-i)**2)/(2*radius) 
            int_offset = int(offset)
            decimal_offset = offset-int_offset
            for j in range(patch.shape[0]):
                if j>int_offset:
                    patch[j,i]=int(decimal_offset*patch_copy[j-1-int_offset,i]+(1-decimal_offset)*patch_copy[j-int_offset,i])
                else:
                    patch[j,i]= np.random.randn() * np.sqrt(3) + 250
        # else:
        #     radius = length**2/(4*height)
        #     for i in range(length):
        #         offset = ((central-i)**2)/(2*radius) 
        #         int_offset = int(offset)
        #         decimal_offset = offset-int_offset
        #         for j in range(patch.shape[0]):
        #             if j>int_offset:
        #                 patch[j,i]=int(decimal_offset*patch_copy[j-1-int_offset,i]+(1-decimal_offset)*patch_copy[j-int_offset,i])
        #             else:
        #                 patch[j,i]= np.random.randn() * np.sqrt(3) + 250            


    return np.uint8(np.round(np.clip(patch,0,255)))

def GetParallel(distribution, height, length, period):
    if length<constant_length: # constant length
        canvas = Gassian((height+2*period,length), mean=255, var = 3)  
    else: # variable length
        canvas = Gassian((height+2*period,length+4*period), mean=255, var = 3)  

    distensce = Gassian((1,int(height/period)+2), mean = period, var = period/5)
    # distensce = Gassian((1,int(height/period)+1), mean = period, var = 0)
    distensce = np.uint8(np.round(np.clip(distensce, period*0.8,period*1.25)))

    begin = 0
    for i in np.squeeze(distensce).tolist():
        newline = Getline(distribution=distribution, length=length)
        h,w = newline.shape
        # cv2.imshow('line', newline)
        # cv2.waitKey(0)
        # cv2.imwrite("D:/ECCV2020/simu_patch/Line3.jpg",newline)

        if begin < height:
            m = np.minimum(canvas[begin:(begin + h),:], newline)
            canvas[begin:(begin + h),:] = m
            begin += i
        else:
            break

    return canvas[:height,:]

def ChooseDistribution(period, Grayscale):
    distribution = np.zeros((period,2))
    c = period/2.0
    difference = 255-Grayscale
    for i in range(distribution.shape[0]):
        distribution[i][0] = Grayscale + difference*abs(i-c)/c*0.5 ########
        distribution[i][1] = np.cos((i-c)/c*(0.5*3.1415929))*difference

        # distribution[i][0] -= np.cos((i-4)/4.0*(0.5*3.1415929))*difference
        # distribution[i][1] += np.cos((i-4)/4.0*(0.5*3.1415929))*difference

    return np.abs(distribution)
    

if __name__ == '__main__':
    
    brush_path = './input/Brush-oil/'+'brush-0.png'
    brush = cv2.imread(brush_path, cv2.IMREAD_GRAYSCALE)
    bgr = [0,192,255]
    color = Gassian_HSV([2,2,3], bgr)

    color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    hsv = [23, 255, 255]
    stroke, mask = GetBrush_HSV(width=1000, length=1600, mean=hsv, brush=brush)
    mask[mask<1] = 0
    white = Gassian_HSV(stroke.shape) # padding   
    mask = mask[:,:,np.newaxis]
    white = np.uint8((1-mask)*white + mask*stroke)
    cv2.imshow('canvas', np.uint8(cv2.cvtColor(white, cv2.COLOR_HSV2BGR)))
    cv2.moveWindow('canvas',1000,100)
    cv2.waitKey(0)
    cv2.imwrite('./input/Brush-oil/'+'color-0.png', cv2.cvtColor(white, cv2.COLOR_HSV2BGR))

    stroke = GetParallel_HSV(width=500, length=300, mean=hsv, grayscale=100)
    
    cv2.imshow('canvas', np.uint8(cv2.cvtColor(stroke, cv2.COLOR_HSV2BGR)))
    cv2.moveWindow('canvas',1000,100)
    cv2.waitKey(0)

    stroke = GetPatch_HSV(num=10, width=16, length=100, SSAA=4, hsv=hsv)
    cv2.imshow('canvas', np.uint8(cv2.cvtColor(stroke, cv2.COLOR_HSV2BGR)))
    cv2.waitKey(0)

    np.random.seed(14)
    # cv2 默认BGR
    canvas = Gassian_HSV(size=(500,500,3), mean=[0,255,50], var = [0,10,0])  # hsv 颜色空间测试
    canvas = cv2.cvtColor(canvas, cv2.COLOR_HSV2BGR)
    # cv2.imshow('bgr', canvas)
    # cv2.waitKey(0)


    period = 40
    mean = [0, 200, 200]
    var = [0,0,0]

    line = Getline_HSV(period=period, length = 500, mean=mean)
    line = cv2.cvtColor(line, cv2.COLOR_HSV2BGR)
    cv2.imshow('line', line[:period])
    cv2.waitKey(0)
    cv2.imwrite("./output/line.png", line[:period])
    # distribution = ChooseDistribution(period=period, Grayscale=Grayscale)
    # print(distribution)
    # patch = GetParallel(distribution=distribution, height=H, length=L, period=period)
    # (h,w) = patch.shape 


    # canvas[255-int(h/2):255-int(h/2)+h,255-int(w/2):255-int(w/2)+w] = patch
    
    # cv2.imshow('Parallel', patch[:, 2*distribution.shape[0]:w-2*distribution.shape[0]])

    # cv2.imshow('Parallel', canvas)
    # cv2.waitKey(0)
    # cv2.imwrite("D:/ACM2022/Debug/Parallel.jpg", patch)


    print("done")



