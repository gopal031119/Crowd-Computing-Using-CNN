from django.shortcuts import render
from .forms import ImageForm
from .models import Image as Img
import base64
from io import BytesIO
import cv2
import h5py
import scipy
import io as IO
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as c
from keras.models import model_from_json
from scipy.ndimage.filters import gaussian_filter
from scipy import spatial
import scipy.io as io 

def load_model():
    # Function to load and return neural network model 
    json_file = open('C:/Users/GOPAL/Desktop/web/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("C:/Users/GOPAL/Desktop/web/model_A_weights.h5")
    return loaded_model
  
model = load_model()

def create_img(path):
    #Function to load,normalize and return image 
    # print(path)
    im = Image.open(path).convert('RGB')
    
    w, h= im.size
    if w>1000 and h>1000:
      im = imageSizeReducer(im)
    
    im = np.array(im)
    print(im.shape)
    # im = np.dot(im[...,:3], [0.2989, 0.5870, 0.1140])
    im = im/255.0
    
    im[:,:,0]=(im[:,:,0]-0.485)/0.229
    im[:,:,1]=(im[:,:,1]-0.456)/0.224
    im[:,:,2]=(im[:,:,2]-0.406)/0.225


    im = np.expand_dims(im,axis  = 0)
    print(im.shape)
    return im


def predict(path):
    #Function to load image,predict heat map, generate count and return (count , image , heat map)
    image = create_img(path)
    ans = model.predict(image)
    print(type(ans))
    print(ans.shape)
    count = np.sum(ans)

    return count,image,scipy.ndimage.filters.gaussian_filter(ans, sigma=0,mode='mirror')


def to_image(numpy_img):
    img = Image.fromarray(numpy_img, 'RGB')
    return img

def to_data_uri(pil_img):
    data = BytesIO()
    pil_img.save(data, "JPEG") # pick your format
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/jpeg;base64,'+data64.decode('utf-8') 

def imageSizeReducer(im):
    # im = Image.open(path).convert('RGB')
    print("Needs to recude size")
    w, h = im.size
    reducing_factor = 1
    if w>1000 and h>1000:
      if w >= h and w > 1000:
        reducing_factor = w/1000
      elif h >= w  and h > 1000:
        reducing_factor = h/1000

    reducing_factor = int(reducing_factor)

    print(im.size,reducing_factor)

    im = im.resize((int(w/reducing_factor),int(h/reducing_factor) ),Image.ANTIALIAS)
    print("After : {}".format(im.size))
  
    plt.imshow(im)
    plt.show()
    return im
  # im = np.array(im)
  # print(im.shape)
  # # im = np.dot(im[...,:3], [0.2989, 0.5870, 0.1140])
  # im = im/255.0
  
  # im[:,:,0]=(im[:,:,0]-0.485)/0.229
  # im[:,:,1]=(im[:,:,1]-0.456)/0.224
  # im[:,:,2]=(im[:,:,2]-0.406)/0.225


  # im = np.expand_dims(im,axis  = 0)
  # print(im.shape)
  # return im

def home(request):
  uploaded_image = "/media/"
  if request.method == "POST":

    form = ImageForm(request.POST, request.FILES)
    if form.is_valid():
      data = Img.objects.filter(photo='myimage/{}'.format(request.FILES['photo']))
      if len(data) == 0:
        # if len(Img.objects.all())>=10 :
        #   Img.objects.all().delete()
        form.save()
        print("Not found")
        # print('myimage/{}'.format(request.FILES['photo']))
        data = Img.objects.filter(photo='myimage/{}'.format(request.FILES['photo']))
        print(data)
      else:
        print("Exists")      

      for el in data:
        uploaded_image = el.photo.url
      print(uploaded_image)

      # imageSizeReducer("C:/Users/kppad/Desktop/Image Upload/imageuploader" + uploaded_image)
      # print(temp.shape)
      # return render(request, 'myapp/home.html', {'form':form})

      ans,im,hmap = predict("C:/Users/GOPAL/Desktop/sem-8/Project/Image Upload/imageuploader" + uploaded_image)

      hmap = hmap.reshape(hmap.shape[1],hmap.shape[2])
      hmap = cv2.resize(hmap,(int(hmap.shape[1]*8),int(hmap.shape[0]*8)),interpolation = cv2.INTER_CUBIC)*8
      # w, h = hmap.shape
      # enlarging_factor = 8
      # hmap = hmap.resize((int(enlarging_factor*w),int(enlarging_factor*h) ),Image.ANTIALIAS)
      rgb = np.stack([hmap]*3, axis=2)
      frm = (rgb * 255 / np.max(rgb)).astype('uint8')
      new = Image.fromarray(frm)

      im_file = BytesIO()
      new.save(im_file, format="PNG")
      im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
      im_b64 = base64.b64encode(im_bytes).decode('utf-8')
      form = ImageForm()

      ans = int(ans)
      
      if ans<=50:
        dtype = "Lightly"
      elif ans<=100:
        dtype = "Moderately"
      else:
        dtype = "Heavily"
      
      return render(request, 'myapp/home.html', {'form':ImageForm(),'uploaded_image':uploaded_image,'density_map':im_b64,'count':ans,'type':dtype})
    
  form = ImageForm()
  return render(request, 'myapp/home.html', {'form':form})
      # form.objects.get_or_create(photo='myimage/{}'.format(request.FILES['photo']))
      # form.save()
      # print(request.FILES['photo'])

  # path = 'C:/Users/kppad/Desktop/Image Upload/imageuploader/media/myimage/diversity-people-group-team-union-260nw-379530769.jpg'

  # print("hmap :")
  # print(hmap.shape)
  # hmap = hmap.reshape(hmap.shape[1],hmap.shape[2])
  # print(hmap)
  # print(hmap.shape)
  # plt.imshow(hmap)
  # plt.show()

  # img = Image.fromarray(hmap, 'RGB')
  # # pil_image = PIL.Image.open('Image.jpg').convert('RGB') 
  # open_cv_image = np.array(img) 
  # # Convert RGB to BGR 
  # open_cv_image = open_cv_image[:, :, ::-1].copy() 
  # img_str = cv2.imencode('.png', open_cv_image)[1].tostring()  # Encode the picture into stream data, put it in the memory cache, and then convert it into string format
  # # print(img_str)
  # b64_code = base64.b64encode(img_str) # Encode into base64
  
  # img = Image.fromarray(hmap, 'RGB')
  # output_buffer = StringIO()
  # img.save(output_buffer, format='PNG')
  # binary_data = output_buffer.getvalue()
  # base64_data = base64.b64encode(binary_data)

  # plt.imshow(frm)
  # plt.show()
      
  # plt.imshow(new)
  # plt.show()
  
      

  # with open(path, "rb") as image_file:
  #   data = image_file.read()
  #   print(data)
  #   image_data = base64.b64encode(data).decode('utf-8')
  # ctx = dict()
  # ctx["image"] = image_data 
