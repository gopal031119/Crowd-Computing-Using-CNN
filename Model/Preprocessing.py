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
        #     sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        # else:
        #     sigma = np.average(np.array(gt.shape))/2./2. # case: 1 point        
        
        # Convolve with the  gaussian filter
        
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma=2.5, mode='constant')    
    
    return density

def display_image(path):
    image = Image.open(path)
    image.show()

root = 'C:/Users/Lenovo/Desktop/Crowd Computing using CSRNet/ShanghaiTech'
part_A_train = os.path.join(root,'part_A/train_data','images')
part_A_test = os.path.join(root,'part_A/test_data','images')
# part_B_train = os.path.join(root,'part_B/train_data','images')
# part_B_test = os.path.join(root,'part_B/test_data','images')
path_sets = [part_A_train,part_A_test]

# List of all image paths
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
print(len(img_paths))

# for i in tqdm(range(5),desc="Images"):
#     display_image(img_paths[i])

img_paths = img_paths[100:]
i = 0
for img_path in tqdm(img_paths):
    # Load sparse matrix
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_'))
    gt = mat["image_info"][0,0][0,0][0]     # to understand this line go to ground_data_file_structure.py
    
    # Read image
    img= plt.imread(img_path)

    # Create a zero matrix of image size
    k = np.zeros((img.shape[0],img.shape[1]))
    
    # Generate hot encoded matrix of sparse matrix
    for i in range(0,len(gt)):  # gt.shape = (277,2)
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1

    # generate density map
    k = gaussian_filter_density(k)
    
    # File path to save density map
    file_path = img_path.replace('.jpg','.h5').replace('images','density_maps')
    
    with h5py.File(file_path, 'w') as hf:
            hf['density'] = k


for img_path in img_paths:

    file_path = img_path.replace('.jpg','.h5').replace('images','density_maps') 
    print(file_path)

    # Sample Ground Truth
    gt_file = h5py.File(file_path,'r')
    groundtruth = np.asarray(gt_file['density'])
    plt.imshow(groundtruth,cmap=CM.jet)
    print("Sum = " ,np.sum(groundtruth))

    file_path = img_path.replace('.jpg','.h5').replace('images','data.csv')
    file_path = file_path[:file_path.index(".csv")+4]

    with open(file_path, 'a', encoding = 'utf-8') as f:
        f.write("{},{},\n".format(img_path.replace('\\','/'),np.sum(groundtruth)))