import cv2
import numpy as np

input_path = './input/Peachl.jpg'
output_path = './input' 

img = cv2.imread(input_path, cv2.IMREAD_COLOR)
# res = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_REPLICATE)
# # res[:,:20,:] = cv2.blur(res[:,:20,:], (10,10))
# # res[:,:20,:] = cv2.GaussianBlur(res[:,:20,:], (9,9), sigmaX=1, sigmaY=1)
# cv2.imshow('res', res)
# cv2.waitKey(0) 
# cv2.imwrite(output_path + "/Leo.jpg", res)
# img = img[1:-1,:,:]

# img = img[:-11,5:-6,:]

min_length = 400
(h,w,c) = img.shape
if h<w:
    img = cv2.resize(img,(min_length,min_length))
else:
    img = cv2.resize(img,(min_length,min_length))
# if h<w:
#     img = cv2.resize(img,(int(min_length*w/h),min_length))
# else:
#     img = cv2.resize(img,(min_length,int(min_length*h/w)))

cv2.imshow('res', img)
cv2.waitKey(0) 
cv2.imwrite(output_path + "/Peach.jpg", img)
