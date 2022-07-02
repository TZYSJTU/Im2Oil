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
        "display": True,
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

    # Plot voronoi regions if you want
    # for region in vor.filtered_regions:
    #     vertices = vor.vertices[region, :]
    #     ax.plot(vertices[:, 0], vertices[:, 1], linewidth=.5, color='.5' )
