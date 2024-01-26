import numpy as np  
import matplotlib.pyplot as plt  
import h5py
from PIL import Image
import matplotlib.image as img 

# IMG_no.h5 file structure :
    # <KeysViewHDF5 ['density']>
    # <HDF5 dataset "density": shape (768, 1024), type "<f4">

image_name = "IMG_12"
dataset = "train_data"  # 'test_data'
part = 'part_A' # 'part_A'

with open('C:/Users/Lenovo/Desktop/Crowd Computing using CSRNet/ShanghaiTech/{}/{}/data.csv'.format(part,dataset), 'r', encoding = 'utf-8') as f1:
    data = f1.readlines()
    for row in data:
        if image_name in row:
            ground_truth = round(float(row.split(',')[1]),3)
            break

normal_image = img.imread('C:/Users/Lenovo/Desktop/Crowd Computing using CSRNet/ShanghaiTech/{}/{}/images/{}.jpg'.format(part,dataset,image_name)) 
plt.subplot(1,2,1)
plt.imshow(normal_image)
plt.title("Normal Image")

f = h5py.File('C:/Users/Lenovo/Desktop/Crowd Computing using CSRNet/ShanghaiTech/{}/{}/density_maps/{}.h5'.format(part,dataset,image_name),'r')
plt.subplot(1,2,2)
plt.imshow(np.asarray(f['density'])+[[20]])     # 'bwr'
print(type(np.asarray(f['density'])))
plt.title("Density map\nground-truth : {}".format(ground_truth))

plt.show()