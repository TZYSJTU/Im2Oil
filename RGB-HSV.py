import cv2
import numpy as np
import math
from simulate_RGB import *

canvas = Gassian_HSV(size=(200,200,3), mean=[120,255,100], var = [0,0,0])  # hsv 颜色空间测试
canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_HSV2BGR)
print(canvas_bgr[0,0])
cv2.imshow('bgr', canvas_bgr)
cv2.waitKey(0)