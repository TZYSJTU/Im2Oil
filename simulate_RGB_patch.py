import cv2
import numpy as np
import math

from numpy.core.fromnumeric import size

constant_length = 1000




# H范围是[0,179](若在180-255，等于减去180)，S范围是[0,255]，V范围是[0,255]
def Gassian_HSV(size, mean=[0,0,255], var=[0,0,0]):
    # size is [h,w,3]
    # mean is [h,s,v]
    # var is  [h,s,v]
    norm = np.random.randn(*size)
    denorm = norm * np.sqrt(var) + mean
    return np.uint8(np.round(np.clip(denorm,0,255)))


def Getline_HSV(period, length, mean, var=[0,0,0]):
    # if length < constant_length:  # if length is shorter than constant_length, lines are Aligned
    # patch = Gassian_HSV((2*period, length,3)) 
    # begin = 0
    # end = 1
    # for i in range(period):
    patch=Gassian((period,length,3), mean, var)
    mask = Getline_Mask_new(period,length,var=1)
    patch[:,:,1:] *= mask

    patch[:,:,2:] += (1-mask)*255
        # print(patch[0])
    # patch[:,:,0] = np.clip(patch[:,:,0],0,180)
    # patch[:period] = Attenuation(patch[:period], period=period,begin=begin, end=end)
    # patch = Distortion(patch, begin=begin, end=end)

    return np.uint8(np.round(np.clip(patch,0,255)))







def Getline_Mask(period, length, var=1):
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

# useless
def Getline_Mask_new(period, length, var=1):
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






def GetPatch_HSV(length, width, mean, var=[0,0,0]):
    # if length[0] < length[1]: 
    #     L = length[1]
    #     W = length[0]
    # else:
    #     W = length[1]
    #     L = length[0]

    patch = Gassian_HSV(size=(width, length,3), mean= mean, var=var) 
    # begin = 0
    # end = 1
    # for i in range(period):
    #     patch[i]=Gassian((1,length,3), mean, var)
    # mask = Getline_Mask(period,length,var=1)
    # patch[:period] *= mask

    # patch[:period,:,2:] += (1-mask)*255

        # print(patch[0])
    # patch[:,:,0] = np.clip(patch[:,:,0],0,180)
    # patch[:period] = Attenuation(patch[:period], period=period,begin=begin, end=end)
    # patch = Distortion(patch, begin=begin, end=end)

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

    patch = Attenuation(patch, period=period, distribution=distribution,begin=begin, end=end)
    patch = Distortion(patch, begin=begin, end=end)

    return np.uint8(np.round(np.clip(patch,0,255)))

def Attenuation(patch, period, begin, end):
    patch = np.float32(patch)
    if period % 2 == 1: # 宽度为奇数
        for i in range (period//2):
            patch[i,:,1] *= 1-(period//2-i)/(period/2)
            patch[-i,:,1] *= 1-(period//2-i)/(period/2)
        patch[period//2,:,1] *= 1-(period//2-i)/(period/2)
    else: # 宽度为偶数
        for i in range (period//2):
            patch[i,:,1] *= 1-(period//2-i)/period
            patch[-i,:,1] *= 1-(period//2-i)/period

    return np.uint8(np.round(np.clip(patch,0,255)))


def Distortion(patch, begin, end):
    # begin 正数第几个像素 end 倒数第几个像素
    height = int(patch.shape[0]/2)
    length = patch.shape[1]
    patch = np.float32(patch)
    patch_copy = patch.copy()

    # central = ((length-begin-end)/2+begin) + np.random.randn()*length/30
    central = ((length-begin-end)/2+begin) # 最中间的像素
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
                patch[j,i]=(decimal_offset*patch_copy[j-1-int_offset,i]+(1-decimal_offset)*patch_copy[j-int_offset,i])
            else:
                patch[j,i,2]= np.random.randn() * np.sqrt(3) + 255
   
    patch_copy = patch.copy()
    if length>100:
        for i in range(length):
            offset = ((central-i)**2)/(2*radius) 
            int_offset = int(offset)
            decimal_offset = offset-int_offset
            for j in range(patch.shape[0]):
                if j>int_offset:
                    patch[j,i]=(decimal_offset*patch_copy[j-1-int_offset,i]+(1-decimal_offset)*patch_copy[j-int_offset,i])
                else:
                    patch[j,i,2]= np.random.randn() * np.sqrt(3) + 255
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
        #                 patch[j,i]= np.random.randn() * np.sqrt(3) + 255            


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
        distribution[i][0] = Grayscale + difference*abs(i-c)/c
        distribution[i][1] = np.cos((i-c)/c*(0.5*3.1415929))*difference

        # distribution[i][0] -= np.cos((i-4)/4.0*(0.5*3.1415929))*difference
        # distribution[i][1] += np.cos((i-4)/4.0*(0.5*3.1415929))*difference

    return np.abs(distribution)
    

if __name__ == '__main__':
    np.random.seed(14)
    # cv2 默认BGR
    canvas = Gassian_HSV(size=(200,500,3), mean=[0,0,255], var = [0,0,0])  # hsv 颜色空间测试
    canvas = cv2.cvtColor(canvas, cv2.COLOR_HSV2BGR)
    # cv2.imshow('bgr', canvas)
    # cv2.waitKey(0)


    period = 30
    mean = [0, 200, 200]
    var = [0,0,0]
    
    begin = 0
    for i in range(4):
        line = Getline_HSV(period=period, length = 500, mean=mean)
        temp = canvas[begin:begin+period] 
        m = (line[:,:,2:]>temp[:,:,2:])
        temp = m*temp + (1-m)*line    
        canvas[begin:begin+period, ::] = temp 
        begin += 10
    canvas = cv2.cvtColor(canvas, cv2.COLOR_HSV2BGR)
    result = canvas[:105]
    cv2.imshow('line', result)
    cv2.waitKey(0)
    cv2.imwrite("./output/patch.png", result)
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



