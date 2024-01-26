import numpy as np
import json
import h5py             # https://pypi.org/project/h5py/
import PIL as image     # https://en.wikipedia.org/wiki/Python_Imaging_Library
import glob
import os
import scipy
from scipy import spatial
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import scipy.io as io 
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import matplotlib.image as mpimg
from PIL import Image

def gaussian_filter_density(gt):

    # Generates a density map using Gaussian filter transformation   
    density = np.zeros(gt.shape, dtype=np.float32)
    
    gt_count = np.count_nonzero(gt)
    
    if gt_count == 0:
        return density
    
    # Find out the K nearest neighbours using a KDTree
   
    pts = np.array(list(zip(np.nonzero(gt)[1].ravel(), np.nonzero(gt)[0].ravel())))
    leafsize = 2048
    
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    
    # query kdtree
    distances, locations = tree.query(pts, k=4)
       
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        # if gt_count > 1:
        #     sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.01
        # else:
        #     sigma = np.average(np.array(gt.shape))/2./2. # case: 1 point        
        
        # Convolve with the gaussian filter
        
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma=2.5  , mode='constant')    
    
    return density

root = 'C:/Users/Lenovo/Desktop/Crowd Computing using CSRNet/ShanghaiTech'
part_B_test = os.path.join(root,'part_B/test_data','images')
# print(type(part_B_train))
path_sets = [part_B_test]

# List of all image paths
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
print(len(img_paths))

#  IMG_251,IMG_305,IMG_284x ...... test_data
img_paths = img_paths[229:230]
i = 0
for img_path in tqdm(img_paths):
    print(img_path)
    # Load sparse matrix
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_'))
    gt = mat["image_info"][0,0][0,0][0]     # to understand this line go to ground_data_file_structure.py
    
    img= plt.imread(img_path)

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Normal Image")

    # Create a zero matrix of image size
    k = np.zeros((img.shape[0],img.shape[1]))
    
    # Generate hot encoded matrix of sparse matrix
    for i in range(0,len(gt)):  # gt.shape = (277,2)
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1

    # generate density map
    k = gaussian_filter_density(k)
    print(np.sum(np.asarray(k)))
    plt.subplot(1,2,2)
    plt.imshow(k,'bwr')
    plt.title("Density map")

    plt.show()
    
    
    # cols = ['Accent', 'Acce1nt_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']
    # RdYIGh
    #bwr_r
    #brg
    # i=1
    # for color in cols[80:90]:
    #     plt.subplot(10,1,i)
    #     plt.title(color)
    #     plt.imshow(k,color)
    #     i += 1
    # plt.show()