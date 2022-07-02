import numpy as np
import cv2, random, time, os, math

from Line_Tools import *
from simulate_RGB import *
from drawpatch import *
from ETF.edge_tangent_flow import *
from quicksort import *

import tqdm, cv2, argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tool_Voronoi import *
import voronoi 

if __name__ == '__main__':
    Debug = False # save density map
    np.random.seed(0)
    default = {
        "filename":"./input/Dawn.jpg",
        "border_copy": 5,  # use 5
        "p_max": 1.0/16,

        "n_iter": 15,
        "k_size": 5,  
        "figsize": 6,
        "pointsize": (8.0, 8.0),
        "display": False,
        "interactive": False,
        "force": True,
        "save": True,        
    }

    description = "Weighted Voronoi Sampler"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--filename', metavar='image filename', type=str, default=default["filename"],
                        help='Input image filename')
    parser.add_argument('--border_copy', metavar='n', type=int, default=default["border_copy"],
                        help='Border replicate')                    
    parser.add_argument('--n_iter', metavar='n', type=int, default=default["n_iter"],
                        help='Number of iterations')
    parser.add_argument('--k_size', metavar='n', type=int, default=default["k_size"],
                        help='Sobel and Mean Filter size')
    parser.add_argument('--p_max', metavar='f', type=float, default=default["p_max"],
                        help='Maximum sampling rate')
    parser.add_argument('--figsize', metavar='w,h', type=int, default=default["figsize"],
                        help='Sampling figure size')
    parser.add_argument('--pointsize', metavar='(min,max)', type=float, nargs=2, default=default["pointsize"],
                        help='Point mix/max size for final display')
    parser.add_argument('--force', action='store_true', default=default["force"],
                        help='Force recomputation')
    parser.add_argument('--save', action='store_true', default=default["save"],
                        help='Save computed points')
    parser.add_argument('--display', action='store_true', default=default["display"],
                        help='Display final result')
    parser.add_argument('--interactive', action='store_true', default=default["interactive"],
                        help='Display intermediate results (slower)')
    args = parser.parse_args()



    filename = args.filename
    file_name = os.path.basename(filename)
    file_name = file_name.split('.')[0]
    print(file_name)

    input_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)   # gray input
    input_img = cv2.copyMakeBorder(input_img, args.border_copy, args.border_copy, args.border_copy, args.border_copy, cv2.BORDER_REPLICATE)

    
    ######## gradient ########
    if True:
        gradient_x = cv2.Sobel(input_img, cv2.CV_32FC1, 1, 0, ksize=args.k_size) 
        gradient_y = cv2.Sobel(input_img, cv2.CV_32FC1, 0, 1, ksize=args.k_size) 
        gradient_magnitude = np.sqrt(gradient_x**2.0 + gradient_y**2.0) 
        # gradient_Gaussian_blur = cv2.GaussianBlur(gradient_magnitude, (5,5), sigmaX=0.5, sigmaY=0.5)
        gradient_mean_blur = cv2.blur(gradient_magnitude, (args.k_size, args.k_size))
        # gradient_norm = cv2.normalize(gradient_mean_blur, None, 0.0, 1.0, cv2.NORM_MINMAX)

        density = gradient_mean_blur # (1.0 - Grayscale_norm) # * gradient_norm
        density = cv2.normalize(density, None, args.p_max/100, args.p_max, cv2.NORM_MINMAX)

        point_num= int(1.0*density.sum())
        print("anchor number:", point_num)
        output_path = "./Output/"+file_name+'-'+str(point_num)
        if not os.path.exists(output_path):
            os.makedirs(output_path)    
        if Debug:
            # cv2.imshow('gradient_norm', np.uint8(gradient_norm*255))
            # density_img = cv2.normalize(density, None, 0.0, 1.0, cv2.NORM_MINMAX)
            density_map = density.astype("float32")
            density_map = density_map/density_map.max()
            density_map = np.uint8(density_map*255)
            cv2.imshow('density_map', density_map)
            cv2.imwrite(output_path + "/density_map.png", 255-density_map[args.border_copy:-args.border_copy, args.border_copy:-args.border_copy])
            cv2.waitKey(0)





    density = density[::-1, :]
    density_P = density.cumsum(axis=1)
    density_Q = density_P.cumsum(axis=1)

    dirname = output_path
    basename = (os.path.basename(filename).split('.'))[0]
    pdf_filename = os.path.join(dirname, basename + "-{}.pdf".format(point_num))
    png_filename = os.path.join(dirname, basename + "-{}.png".format(point_num))
    dat_filename = os.path.join(dirname, basename + "-{}.npy".format(point_num))

    # Initialization
    if not os.path.exists(dat_filename) or args.force:
        points = initialization(point_num, density)
        print("Nb points:", point_num)
        print("Nb iterations:", args.n_iter)
    else:
        points = np.load(dat_filename)
        print("Nb points:", len(points))
        print("Nb iterations: -")
    # print("Density file: %s (resized to %dx%d)" % (
    #       filename, density.shape[1], density.shape[0]))
    # print("Output file (PDF): %s " % pdf_filename)
    # print("            (PNG): %s " % png_filename)
    # print("            (DAT): %s " % dat_filename)

        
    xmin, xmax = 0, density.shape[1]
    ymin, ymax = 0, density.shape[0]
    bbox = np.array([xmin, xmax, ymin, ymax])
    ratio = (xmax-xmin)/(ymax-ymin)

    # Interactive display
    if args.interactive:

        # Setup figure
        fig = plt.figure(figsize=(args.figsize, args.figsize/ratio),
                         facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim([xmin, xmax])
        ax.set_xticks([])
        ax.set_ylim([ymin, ymax])
        ax.set_yticks([])
        scatter = ax.scatter(points[:, 0], points[:, 1], s=1,
                             facecolor="k", edgecolor="None")

        def update(frame):
            global points
            # Recompute weighted centroids
            regions, points = voronoi.centroids(points, density, density_P, density_Q)

            # Update figure
            Pi = points.astype(int)
            X = np.maximum(np.minimum(Pi[:, 0], density.shape[1]-1), 0)
            Y = np.maximum(np.minimum(Pi[:, 1], density.shape[0]-1), 0)
            sizes = (args.pointsize[0] +
                     (args.pointsize[1]-args.pointsize[0])*density[Y, X])
            scatter.set_offsets(points)
            scatter.set_sizes(sizes)
            bar.update()

            # Save result at last frame
            if (frame == args.n_iter-2 and
                      (not os.path.exists(dat_filename) or args.save)):
                np.save(dat_filename, points)
                plt.savefig(pdf_filename)
                plt.savefig(png_filename)

        bar = tqdm.tqdm(total=args.n_iter)
        animation = FuncAnimation(fig, update,
                                  repeat=False, frames=args.n_iter-1)
        plt.show()

    elif not os.path.exists(dat_filename) or args.force:
        for i in tqdm.trange(args.n_iter):
            regions, points = voronoi.centroids(points, density, density_P, density_Q)

            
    if (args.save or args.display) and not args.interactive:
        fig = plt.figure(figsize=(args.figsize, args.figsize/ratio),
                         facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim([xmin, xmax])
        ax.set_xticks([])
        ax.set_ylim([ymin, ymax])
        ax.set_yticks([])
        scatter = ax.scatter(points[:, 0], points[:, 1], s=1, 
                             facecolor="k", edgecolor="None")
        Pi = points.astype(int)
        X = np.maximum(np.minimum(Pi[:, 0], density.shape[1]-1), 0)
        Y = np.maximum(np.minimum(Pi[:, 1], density.shape[0]-1), 0)
        sizes = (args.pointsize[0] +
                 (args.pointsize[1]-args.pointsize[0])*density[Y, X])
        scatter.set_offsets(points)
        scatter.set_sizes(sizes)

        # Save stipple points and tippled image
        if not os.path.exists(dat_filename) or args.save:
            np.save(dat_filename, points)
            # plt.savefig(pdf_filename)
            plt.savefig(png_filename)

        if args.display:
            plt.show()




    # file paths
    input_path = args.filename
    file_name = os.path.basename(input_path)
    file_name = file_name.split('.')[0]
    point_path = './output/'+file_name+"-"+str(point_num)
    brush_path = './input/Brush-oil/'+'brush-0.png'
    brush = cv2.imread(brush_path, cv2.IMREAD_GRAYSCALE)

    # config parameter
    np.random.seed(0)
    SSAA = 8                                # SSAA                    
    FREQ = 1000                             # save one frame every（Freq) strokes drawn
    RANDOM_ORDER = False
    USER_ORDER = True   
    DEBUG = False

    # main parameter
    RATIO = 3                               # max_length/max_width
    sample_max = args.p_max                 # maximum sample rate <= 1

    # auto parameter (before SSAA)
    sample_min = sample_max/100             # minimum sample rate
    max_width = int(np.sqrt(1/sample_min))
    max_length = RATIO*max_width            
    min_width = np.sqrt(1/sample_max)-1                            
    min_length = RATIO*min_width                      
                    
    # default parameter
    BORDERCOPY = 5                  # BORDERCOPY
    threshold_hsv  = (30,None,15)   # threshold for hsv
    kernel_sobel = 5                # for sobel
    kernel_radius = 5               # for ETF
    iter_time = 15                  # for ETF
    background_dir = None           # for ETF 



    def Search_Stroke(anchors, density, img_gray, input_hsv, gradient_norm, 
        angle, angle_hatch, min_width, min_length, max_width, max_length, threshold_hsv):  

        (h0,w0) = img_gray.shape
        patch_sequence = []
        for i in tqdm.tqdm(range(anchors.shape[0])):
            patch_temp = {}
            # get coordinate 
            (x,y) = anchors[i]
            y = h0 - y         # y is Upside-down ！！！
            y = np.clip(y,0,h0-1)
            x = np.clip(x,0,w0-1)
            x = int(round(x))  # for savety
            y = int(round(y))  # for savety
            

            patch_temp['coordinate'] =  [y,x]

            patch_temp['angle_hatch'] = angle_hatch[y,x]
            patch_temp['path_hatch'] = FindPath_V7(img_gray, input_hsv, angle_hatch, y, x, step_length=1, min_length=min_width,max_length=max_width, threshold_hsv= threshold_hsv)
            patch_temp['w1'] = patch_temp['path_hatch'].index(patch_temp['coordinate'])
            patch_temp['w2'] = len(patch_temp['path_hatch'])-patch_temp['w1']-1

            patch_temp['angle_ETF'] = angle[y,x]
            patch_temp['path_ETF'] = FindPath_V7(img_gray, input_hsv, angle, y, x, step_length=1, min_length=min_length,max_length=max_length, threshold_hsv= threshold_hsv)
            patch_temp['l1'] = patch_temp['path_ETF'].index(patch_temp['coordinate'])
            patch_temp['l2'] = len(patch_temp['path_ETF'])-patch_temp['l1']-1
        
            patch_temp['grayscale'] = img_gray[y,x]
            patch_temp['hsv'] = input_hsv[y,x]

            patch_temp['gradient'] = gradient_norm[y,x]
            patch_temp['density'] = density[y,x]

            max_local_width = int(np.around(np.sqrt(1.0/patch_temp['density'])))
            patch_temp['w1'] = min(patch_temp['w1'], max_local_width)
            patch_temp['w1'] = max(patch_temp['w1'], min_width)
            patch_temp['w2'] = min(patch_temp['w2'], max_local_width)
            patch_temp['w2'] = max(patch_temp['w2'], min_width)
            patch_temp['l1'] = min(patch_temp['l1'], RATIO*max_local_width)
            patch_temp['l1'] = max(patch_temp['l1'], min_length)
            patch_temp['l2'] = min(patch_temp['l2'], RATIO*max_local_width)
            patch_temp['l2'] = max(patch_temp['l2'], min_length)

            patch_temp['importance'] = (patch_temp['w1']+patch_temp['w2']+1)*(patch_temp['l1']+patch_temp['l2']+1) + np.random.randn()/100

            patch_sequence.append(patch_temp)

        return patch_sequence

    def Render_Stroke(patch_sequence, img_gray, output_path, max_length, save=True):
        
        (h0,w0) = img_gray.shape
        Canvas = Gassian_HSV((h0*SSAA+2*max_length*SSAA,w0*SSAA+2*max_length*SSAA,3)) # padding max_length      
        Mask = np.zeros((h0*SSAA+2*max_length*SSAA,w0*SSAA+2*max_length*SSAA))
        for step in tqdm.tqdm(range(len(patch_sequence))):
            patch_temp = patch_sequence[step]
            w1 = patch_temp['w1'] 
            w2 = patch_temp['w2']              
            l1 = patch_temp['l1']
            l2 = patch_temp['l2']
            length = int(round((l1+1+l2)*SSAA))
            width  = int(round((w1+1+w2)*SSAA))

            angle_ETF = patch_temp['angle_ETF']
            angle_Hatch = angle_Hatch = patch_temp['angle_hatch']

            # get patch central coordinate 
            central = patch_temp['coordinate']
            central[1]-= (l1-l2)/2*math.cos(angle_ETF/180*math.pi) # x
            central[0]+= (l1-l2)/2*math.sin(angle_ETF/180*math.pi) # y
            central[1]-= (w1-w2)/2*math.cos(angle_Hatch/180*math.pi) # x
            central[0]+= (w1-w2)/2*math.sin(angle_Hatch/180*math.pi) # y


            Y, X = int(round(central[0]*SSAA)), int(round(central[1]*SSAA))  
            Y += max_length*SSAA    # Canvas edge is padded
            X += max_length*SSAA    # Canvas edge is padded 

        
            hsv = patch_temp['hsv']

            stroke, mask = GetBrush_HSV(width=width, length=length, mean=hsv, brush=brush)
            rotate_stroke, _ = rotate_hsv(stroke, angle_ETF, pad_color=(int(hsv[0]),0,255))
            rotate_stroke = np.clip(rotate_stroke,0,255)
            (rotate_h, rotate_w, _) = rotate_stroke.shape 
            rotate_hh, rotate_ww = int(rotate_h/2), int(rotate_w/2)

            mask, _ = rotate(mask, angle_ETF, pad_color=0)
            mask[mask<1] = 0
            # cv2.imshow('Canvas', np.uint8(rotate_stroke))
            # cv2.waitKey(0)



            Mask[Y-rotate_hh:Y-rotate_hh+rotate_h, X-rotate_ww:X-rotate_ww+rotate_w] += mask
            temp = Canvas[Y-rotate_hh:Y-rotate_hh+rotate_h, X-rotate_ww:X-rotate_ww+rotate_w] 

            rotate_alpha = mask.astype("float32")
            rotate_alpha = rotate_alpha[:,:,np.newaxis]
            rotate_stroke = rotate_stroke.astype("float32")

            # white = Gassian_HSV(rotate_stroke.shape) # padding   
            # white = np.uint8((1-rotate_alpha)*white + rotate_alpha*rotate_stroke)
            # cv2.imwrite(output_path + "/stroke/{0:05d}.png".format(step), cv2.cvtColor(white, cv2.COLOR_HSV2BGR))
            Canvas[Y-rotate_hh:Y-rotate_hh+rotate_h, X-rotate_ww:X-rotate_ww+rotate_w] = np.uint8((1-rotate_alpha)*temp + rotate_alpha*rotate_stroke)

            if save and (step+1) % FREQ == 0:
                result = Canvas[max_length*SSAA:-max_length*SSAA,max_length*SSAA:-max_length*SSAA] 
                cv2.imwrite(output_path + "/process/{0:05d}.png".format(step+1), cv2.cvtColor(result[BORDERCOPY*SSAA:-BORDERCOPY*SSAA,BORDERCOPY*SSAA:-BORDERCOPY*SSAA], cv2.COLOR_HSV2BGR))
                # cv2.imshow('step', Canvas)
                # cv2.waitKey(0) 
        if save and step == len(patch_sequence)-1 and step+1 % FREQ != 0: # 存最后一张
            result = Canvas[max_length*SSAA:-max_length*SSAA,max_length*SSAA:-max_length*SSAA] 
            cv2.imwrite(output_path + "/process/{0:05d}.png".format(step+1), cv2.cvtColor(result[BORDERCOPY*SSAA:-BORDERCOPY*SSAA,BORDERCOPY*SSAA:-BORDERCOPY*SSAA], cv2.COLOR_HSV2BGR)) 

        Mask[Mask>0] = 1
        # cv2.imshow('mask', np.uint8(Mask*255))
        # cv2.waitKey(0)
        return Canvas, Mask # hsv




    ####### directory #######
    if 1:
        output_path = './output' 
        file_name = os.path.basename(point_path)
        file_name = file_name.split('.')[0]
        print(file_name)
        output_path = output_path+"/"+file_name+"-oil"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            os.makedirs(output_path+"/stroke")
            os.makedirs(output_path+"/process")

    ####### save input ####### 
    if 1:
        input_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)        # bgr输入
        cv2.imwrite(output_path + "/input_bgr.png", input_bgr)
        input_bgr = cv2.copyMakeBorder(input_bgr, BORDERCOPY, BORDERCOPY, BORDERCOPY, BORDERCOPY, cv2.BORDER_REPLICATE)
        input_hsv = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2HSV)      # hsv输入
        input_gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)    # gray输入
        (H0,W0) = input_gray.shape
        # input_bgr = cv2.GaussianBlur(input_bgr, (5,5), sigmaX=0.5, sigmaY=0.5)
        # cv2.imshow('input_bgr', input_bgr)
        # cv2.waitKey(0)    
        

    ####### ETF #######
    if 1:
        time_start=time.time()
        ETF_filter = ETF(img=input_gray, output_path=output_path+'/mask',
            kernel_radius=kernel_radius, iter_time=iter_time, background_dir=background_dir)
        angle = ETF_filter.forward().numpy()
        angle_hatch = angle+90
        angle_hatch[angle_hatch>90] -= 180

        print('ETF done')


    img_gray = input_gray
    ########### density #############
    if 1: 
        x_der = cv2.Sobel(input_gray, cv2.CV_32FC1, 1, 0, ksize=kernel_sobel) 
        y_der = cv2.Sobel(input_gray, cv2.CV_32FC1, 0, 1, ksize=kernel_sobel) 

        gradient_magnitude = np.sqrt(x_der**2.0 + y_der**2.0) 
        gradient_norm = gradient_magnitude/gradient_magnitude.max()
 
        # gradient_Gaussian_blur = cv2.GaussianBlur(gradient_magnitude, (5,5), sigmaX=0.5, sigmaY=0.5)
        gradient_mean_blur = cv2.blur(gradient_magnitude, (kernel_sobel,kernel_sobel))
        gradient_mean_norm = cv2.normalize(gradient_mean_blur, None, 0.0, 1.0, cv2.NORM_MINMAX)

        density = cv2.normalize(gradient_mean_blur, None, sample_min, sample_max, cv2.NORM_MINMAX)
        if DEBUG == True:
            cv2.imshow('gradient_map', np.uint8(gradient_norm*255))
            cv2.imshow('density_map', np.uint8(gradient_mean_norm*255))
            cv2.waitKey(0)       


    ############   Search patch  ##########
    if 1:
        time_start=time.time() # time
        points = np.load(point_path+'/'+file_name+'.npy' ) # stipple coordinates
        patch_sequence = Search_Stroke(points, density, img_gray,input_hsv,gradient_norm,
            angle, angle_hatch, min_width, min_length, max_width, max_length, threshold_hsv)
        print('Stoke Searching time', int(time.time()-time_start),"seconds")
        print('Stoke number', len(patch_sequence))



    ############   Stroke Order   ##########
    if RANDOM_ORDER:
        random.shuffle(patch_sequence)   
    elif USER_ORDER == True:
            random.shuffle(patch_sequence)   
            quickSort(patch_sequence,0,len(patch_sequence)-1)
    

    ############   Render Stroke  ##########
    if 1:
        time_start=time.time() # time
        wihte = Gassian_HSV((H0*SSAA-2*BORDERCOPY*SSAA,W0*SSAA-2*BORDERCOPY*SSAA,3)) # padding   
        cv2.imwrite(output_path + "/process/{0:04d}.png".format(0), cv2.cvtColor(wihte, cv2.COLOR_HSV2BGR))

        Canvas, Mask = Render_Stroke(patch_sequence, img_gray, output_path, max_length, save=True)
        print('Stoke Rendering time', int(time.time()-time_start),"seconds")
        print('Stoke number', len(patch_sequence))

        result = Canvas[max_length*SSAA:-max_length*SSAA,max_length*SSAA:-max_length*SSAA]
        cv2.imwrite(output_path + "/Oil_drawing.png", cv2.cvtColor(result, cv2.COLOR_HSV2BGR))


    ########### Pad Blank Area ###########

    mask = Mask
    mask[ max_length*SSAA+BORDERCOPY*SSAA-1,:] = 1
    mask[-max_length*SSAA-BORDERCOPY*SSAA,:] = 1
    mask[:, max_length*SSAA+BORDERCOPY*SSAA-1] = 1
    mask[:,-max_length*SSAA-BORDERCOPY*SSAA] = 1
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
            if p[0] >= BORDERCOPY*SSAA and p[1] >= BORDERCOPY*SSAA and p[0] < result.shape[1]-BORDERCOPY*SSAA and p[1] < result.shape[0]-BORDERCOPY*SSAA and stats[i][4]>0 and stats[i][4]<result.shape[0]*result.shape[1]/4:
                p[0], p[1] = p[0]/SSAA, p[1]/SSAA
                Points.append([p[0],result.shape[0]/SSAA-p[1]]) # x, y
        
        Points = np.array(Points)
        if Points.shape[0] == 0:
            cv2.imwrite(output_path + "/Final_Result.png", result[BORDERCOPY*SSAA:-BORDERCOPY*SSAA,BORDERCOPY*SSAA:-BORDERCOPY*SSAA])
            cv2.imwrite(output_path + "/process/Final_Result.png", result[BORDERCOPY*SSAA:-BORDERCOPY*SSAA,BORDERCOPY*SSAA:-BORDERCOPY*SSAA])
            break
        else:
            for point in Points:
                cv2.circle(result, (int(np.around(point[0]*SSAA)),int(np.around((result.shape[0]/SSAA-point[1])*SSAA))), 3, (0,0,255), 3)
            cv2.imwrite(output_path + "/anchor.png", result)     

            ####  Search ###
            pad_sequence = Search_Stroke(np.array(Points), density, img_gray, input_hsv, gradient_norm,
                angle, angle_hatch, min_width, min_length, max_width, max_length, threshold_hsv)
            if USER_ORDER == True:
                quickSort(pad_sequence,0,len(pad_sequence)-1)
            ### Pad ###
            pad_canvas, pad_mask = Render_Stroke(pad_sequence, img_gray, output_path, max_length, save=False)
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

        min_length += RATIO
        min_width += 1    
