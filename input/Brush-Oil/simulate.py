import cv2
import numpy as np

# H范围是[0,179](若在180-255，等于减去180)，S范围是[0,255]，V范围是[0,255]
def Gassian_HSV(size, mean=[0,0,255], var=[0,0,0]):
    # size is [h,w,3]
    # mean is [h,s,v]
    # var is  [h,s,v]
    norm = np.random.randn(*size)
    denorm = norm * np.sqrt(var) + mean
    return np.uint8(np.round(np.clip(denorm,0,255)))


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

brush_path = './brush-3.png'
brush = cv2.imread(brush_path, cv2.IMREAD_GRAYSCALE)

bgr = [0, 192, 255]
color = Gassian_HSV(size=(100,100,3), mean=bgr, var = [0,0,0])
color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
cv2.imshow('color', np.uint8(cv2.cvtColor(color, cv2.COLOR_HSV2BGR)))
cv2.moveWindow('canvas',1000,100)
cv2.waitKey(0)

stroke, mask = GetBrush_HSV(width=200, length=400, mean=[23,255,255], brush=brush)
cv2.imshow('canvas', np.uint8(cv2.cvtColor(stroke, cv2.COLOR_HSV2BGR)))
cv2.moveWindow('canvas',1000,100)
cv2.waitKey(0)