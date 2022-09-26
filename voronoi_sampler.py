import tqdm, cv2, argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from voronoi_tools import *
import voronoi 

def K_Means_Sampler(output_dir, filename, p_max, p_min, border_copy=5, k_size=5, n_iter=15,
    figsize=6, pointsize= (8.0, 8.0), display=True, force=True, save=True):


    input_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)   # gray input
    input_img = cv2.copyMakeBorder(input_img, border_copy, border_copy, border_copy, border_copy, cv2.BORDER_REPLICATE)

    ######## gradient ########
    if True:
        gradient_x = cv2.Sobel(input_img, cv2.CV_32FC1, 1, 0, ksize=k_size) 
        gradient_y = cv2.Sobel(input_img, cv2.CV_32FC1, 0, 1, ksize=k_size) 
        gradient_magnitude = np.sqrt(gradient_x**2.0 + gradient_y**2.0) 
        density = cv2.blur(gradient_magnitude, (k_size, k_size))
        density = cv2.normalize(density, None, p_min, p_max, cv2.NORM_MINMAX)

        point_num= int(density.sum())
        print("anchor number:", point_num)
        # output_path = output_dir+'/'+str(point_num)

        if save:
            density_map = density.astype("float32")
            density_map = density_map/density_map.max()
            density_map = np.uint8(density_map*255)
            cv2.imwrite(output_dir + "/density_map.png", 255-density_map[border_copy:-border_copy, border_copy:-border_copy])
            if display == True:
                cv2.imshow('density_map', density_map)
                cv2.waitKey(0)



    density_ = density[::-1, :]
    density_P = density_.cumsum(axis=1)
    density_Q = density_P.cumsum(axis=1)

    dirname = output_dir
    basename = (os.path.basename(filename).split('.'))[0]
    png_filename = os.path.join(dirname, basename + "-{}.png".format(point_num))
    dat_filename = os.path.join(dirname, basename + "-{}.npy".format(point_num))

    # Initialization
    if not os.path.exists(dat_filename) or force:
        points = initialization(point_num, density_)
        print("Nb points:", point_num)
        print("Nb iterations:", n_iter)
    else:
        points = np.load(dat_filename)
        print("Nb points:", len(points))
        print("Nb iterations: -")


    xmin, xmax = 0, density_.shape[1]
    ymin, ymax = 0, density_.shape[0]
    ratio = (xmax-xmin)/(ymax-ymin)


    if not os.path.exists(dat_filename) or force:
        for i in tqdm.trange(n_iter):
            regions, points = voronoi.centroids(points, density_, density_P, density_Q)

            
    if save or display:
        fig = plt.figure(figsize=(figsize, figsize/ratio),
                         facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim([xmin, xmax])
        ax.set_xticks([])
        ax.set_ylim([ymin, ymax])
        ax.set_yticks([])
        scatter = ax.scatter(points[:, 0], points[:, 1], s=1, 
                             facecolor="k", edgecolor="None")
        Pi = points.astype(int)
        X = np.maximum(np.minimum(Pi[:, 0], density_.shape[1]-1), 0)
        Y = np.maximum(np.minimum(Pi[:, 1], density_.shape[0]-1), 0)
        sizes = (pointsize[0] +
                 (pointsize[1]-pointsize[0])*density_[Y, X])
        scatter.set_offsets(points)
        scatter.set_sizes(sizes)

        # Save stipple points and tippled image
        if not os.path.exists(dat_filename) or save:
            np.save(dat_filename, points)
            plt.savefig(png_filename)

        if display:
            plt.show()

    return point_num, density, gradient_magnitude, dat_filename
