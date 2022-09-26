import numpy as np
import cv2, random, time, os, argparse

from simulate_RGB import *
from drawpatch import *
from ETF.edge_tangent_flow import *
from quicksort import * 
from voronoi_sampler import K_Means_Sampler
from search_and_render import *

if __name__ == '__main__':

    default = {
        # config parameters
        "image":"./input/S1.jpg",           # input image filepath
        "brush":"./brush/brush-0.png",      # brush template
        "p_max": 1.0/36,                     # maximum Sampling Rate, try to use 1/4, 1/9, 1/16, 1/25, 1/36
        "seed": 0,                          # np.random.seed()
        "force": True,                      # force recomputation of the anchor Map
        "SSAA" : 8,                         # Super-Sampling Anti-Aliasing                    
        "freq" : 1000,                       # save one frame every（freq) strokes drawn
        "stroke_order_type": 0,             # use 0 for the default size order, use 1 for random order

        # default parameters
        "padding": 5,                       # padding
        "n_iter": 15,                       # K-means iteration
        "k_size": 5,                        # Sobel and Mean Filter size
        "figsize": 6,                       # anchor map figure size
        "pointsize": (8.0, 8.0),            # point (mix,max) size for the anchor map
        "ratio" : 3,                        # max_length/max_width     
        "threshold_hsv": (30,None,15),      # threshold for hsv color space during searching
        "kernel_radius" : 5,                # ETF kernel_radius
        "ETF_iter" : 15,                    # ETF iteration number
        "background_dir" : None,            # for ETF 
    }


    # auto parameters (before SSAA)
    p_max = default["p_max"]                    # maximum sampling rate
    p_min = p_max/100                           # minimum sampling rate
    ratio = default["ratio"]                    # max_length/max_width 
    max_width = np.sqrt(1/p_min)                # maximum stroke width
    min_width = np.sqrt(1/p_max)-1              # minimum stroke width
    max_length = int(ratio * max_width)         # maximum stroke length                       
    min_length = ratio * min_width              # minimum stroke length
    padding = default["padding"]                # padding



    description = "Im2Oil: Stroke-Based Oil Painting Rendering with Linearly Controllable Fineness Via Adaptive Sampling"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--f', type=str, default=default["image"],
                        help='input image path')
    parser.add_argument('--b', type=str, default=default["brush"],
                        help='brush template path')
    parser.add_argument('--p', type=float, default=default["p_max"],
                        help='maximum sampling rate')
    parser.add_argument('--s', type=int, default=default["seed"],
                        help='np.random.seed()')                  
    parser.add_argument('--force', action='store_true', default=default["force"],
                        help='Force recomputation of the Anchor Map')
    parser.add_argument('--SSAA', type=int, default=default["SSAA"],
                        help='Super-Sampling Anti-Aliasing')
    parser.add_argument('--freq', type=int, default=default["freq"],
                        help='save one frame every (freq) strokes drawn')  
    parser.add_argument('--order', type=int, default=default["stroke_order_type"],
                        help='0 for default size order, 1 for random order')  
    args = parser.parse_args()

    
    ####### make directory #######
    if 1:
        filename = os.path.basename(args.f)
        print("filename:", filename)
        filename = filename.split('.')[0]
        # point_path = './output/'+filename+"-"+str(p_max)+"/"+filename+'-'+str(point_num)+".npy"
        brush_path = args.b
        brush = cv2.imread(brush_path, cv2.IMREAD_GRAYSCALE)
        output_path = './output/'+filename+'-p-'+str(int(1/p_max))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            os.makedirs(output_path+"/anchor")
            os.makedirs(output_path+"/stroke")
            os.makedirs(output_path+"/process")



    ######## K-means ########
    np.random.seed(args.s)
    point_num, density, gradient_magnitude, point_path =  \
    K_Means_Sampler(output_dir=output_path+"/anchor", filename=args.f, p_max=p_max, p_min=p_min, 
        border_copy=padding, k_size=default["k_size"], n_iter=default["n_iter"], figsize=default["figsize"], pointsize=default["pointsize"], 
        display=False, force=args.force, save=True)



    ####### save input ####### 
    if 1:
        input_bgr = cv2.imread(args.f, cv2.IMREAD_COLOR)        # bgr输入
        cv2.imwrite(output_path + "/input_bgr.png", input_bgr)
        input_bgr = cv2.copyMakeBorder(input_bgr, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
        input_hsv = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2HSV)      # hsv输入
        input_gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)    # gray输入
        (H0,W0) = input_gray.shape
        

    ####### ETF #######
    if 1:
        time_start=time.time()
        ETF_filter = ETF(img=input_gray, output_path=output_path+'/mask',
            kernel_radius=kernel_radius, iter_time=default["ETF_iter"], background_dir=default["background_dir"])
        angle = ETF_filter.forward().numpy()
        angle_hatch = angle+90
        angle_hatch[angle_hatch>90] -= 180
        print('ETF Filtering time:', int(time.time()-time_start),"seconds")
        print('ETF done')

 


    ############   Search patch  ##########
    if 1:
        time_start=time.time() # time
        points = np.load(point_path) # stipple coordinates
        patch_sequence = Search_Stroke(points, density, input_gray, input_hsv, gradient_magnitude, ratio,
            angle, angle_hatch, min_width, min_length, max_width, max_length, default["threshold_hsv"])
        print('Stoke Searching time', int(time.time()-time_start),"seconds")
        print('Stoke number', len(patch_sequence))



    ############   Stroke Order   ##########
    if args.order == 1:
        random.shuffle(patch_sequence)   
    elif args.order == 0:
        random.shuffle(patch_sequence)   
        quickSort(patch_sequence,0,len(patch_sequence)-1)
    

    ############   Render Stroke  ##########
    if 1:
        SSAA = args.SSAA
        freq = args.freq
        time_start=time.time() # time
        wihte = Gassian_HSV((H0*SSAA-2*padding*SSAA,W0*SSAA-2*padding*SSAA,3)) # padding   
        cv2.imwrite(output_path + "/process/{0:04d}.png".format(0), cv2.cvtColor(wihte, cv2.COLOR_HSV2BGR))

        Canvas, Mask = Render_Stroke(brush, patch_sequence, input_gray, output_path, max_length, SSAA=SSAA, BORDERCOPY=padding, FREQ=freq, save=True)
        print('Stoke Rendering time', int(time.time()-time_start),"seconds")
        print('Stoke number', len(patch_sequence))

        result = Canvas[max_length*SSAA:-max_length*SSAA,max_length*SSAA:-max_length*SSAA]
        cv2.imwrite(output_path + "/Oil_drawing.png", cv2.cvtColor(result, cv2.COLOR_HSV2BGR))


    ########### Pad Blank Area ###########
    if 1:
        mask = Mask
        mask[ max_length*SSAA+padding*SSAA-1,:] = 1
        mask[-max_length*SSAA-padding*SSAA,:] = 1
        mask[:, max_length*SSAA+padding*SSAA-1] = 1
        mask[:,-max_length*SSAA-padding*SSAA] = 1
        # cv2.imshow('mask', np.uint8(mask*255))
        # cv2.waitKey(0)
        while(1):
            result = cv2.imread(output_path + "/Oil_drawing.png", cv2.IMREAD_COLOR) # label输入      
            mask_cut = mask[max_length*SSAA:-max_length*SSAA,max_length*SSAA:-max_length*SSAA]
            cv2.imwrite(output_path + "/mask.png", mask_cut.astype("uint8")*255) 

            connect_num, labels, stats, centroids = cv2.connectedComponentsWithStats(255-mask_cut.astype("uint8")*255, connectivity=8)

            Points = []
            for i in range(centroids.shape[0]):
                p = centroids[i]
                if p[0] >= padding*SSAA and p[1] >= padding*SSAA and p[0] < result.shape[1]-padding*SSAA and p[1] < result.shape[0]-padding*SSAA and stats[i][4]>0 and stats[i][4]<result.shape[0]*result.shape[1]/4:
                    p[0], p[1] = p[0]/SSAA, p[1]/SSAA
                    Points.append([p[0],result.shape[0]/SSAA-p[1]]) # x, y
            
            Points = np.array(Points)
            if Points.shape[0] == 0:
                cv2.imwrite(output_path + "/Final_Result.png", result[padding*SSAA:-padding*SSAA,padding*SSAA:-padding*SSAA])
                cv2.imwrite(output_path + "/process/Final_Result.png", result[padding*SSAA:-padding*SSAA,padding*SSAA:-padding*SSAA])
                break
            else:
                for point in Points:
                    cv2.circle(result, (int(np.around(point[0]*SSAA)),int(np.around((result.shape[0]/SSAA-point[1])*SSAA))), 3, (0,0,255), 3)
                cv2.imwrite(output_path + "/anchor.png", result)     

                ####  Search ###
                pad_sequence = Search_Stroke(np.array(Points), density, input_gray, input_hsv, gradient_magnitude, ratio,
                    angle, angle_hatch, min_width, min_length, max_width, max_length, default["threshold_hsv"])
                if args.order == 0:
                    quickSort(pad_sequence,0,len(pad_sequence)-1)
                ### Pad ###
                pad_canvas, pad_mask = Render_Stroke(brush, pad_sequence, input_gray, output_path, max_length, SSAA=SSAA, BORDERCOPY=padding, FREQ=freq, save=False)
                pad_canvas_cut = pad_canvas[max_length*SSAA:-max_length*SSAA,max_length*SSAA:-max_length*SSAA]
                pad_canvas_cut = cv2.cvtColor(pad_canvas_cut, cv2.COLOR_HSV2BGR)

                for point in Points:
                    cv2.circle(pad_canvas_cut, (int(np.around(point[0]*SSAA)),int(np.around((result.shape[0]/SSAA-point[1])*SSAA))), 3, (0,0,255), 3)
                cv2.imwrite(output_path + "/pad_canvas_cut.png", pad_canvas_cut)    



                Oil_drawing = cv2.imread(output_path + "/Oil_drawing.png", cv2.IMREAD_COLOR) # label输入   
                Oil_drawing = cv2.cvtColor(Oil_drawing, cv2.COLOR_BGR2HSV)
                pad_canvas_cut = pad_canvas[max_length*SSAA:-max_length*SSAA,max_length*SSAA:-max_length*SSAA]
                m = pad_mask*(1-mask)
                m = m[max_length*SSAA:-max_length*SSAA,max_length*SSAA:-max_length*SSAA,np.newaxis]
                Oil_drawing = np.uint8(m*pad_canvas_cut + (1-m)*Oil_drawing)
                Oil_drawing = cv2.cvtColor(Oil_drawing, cv2.COLOR_HSV2BGR)
                cv2.imwrite(output_path + "/Oil_drawing.png", Oil_drawing)  

                mask += pad_mask
                mask[mask>0]=1
                # result[:,:,2] = np.uint8(np.around(np.clip(result[:,:,2].astype("float32")/0.85,0,255)))

            min_length += ratio
            min_width += 1    
