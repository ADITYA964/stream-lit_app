import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from tensorflow.keras import backend as K
from skimage.transform import resize
import os
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Concatenate, Conv2D, Input, Flatten, MaxPooling2D, UpSampling2D, concatenate, Cropping2D, Reshape, BatchNormalization
from keras.layers import Dense, concatenate
from keras.models import Model, load_model
from keras.applications.nasnet import NASNetLarge
from keras.applications import VGG16, VGG19
import tensorflow as tf
import numpy as np
import openslide
from openslide import open_slide, __library_version__ as openslide_version
from skimage.color import rgb2gray
import cv2
import os
import random
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import sys
from skimage.transform import resize
import PIL
import tensorflow as tf
import numpy as np
from openslide import open_slide, __library_version__ as openslide_version
from skimage.color import rgb2gray
import cv2
import os
import random
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from PIL import Image
from openslide.lowlevel import *
from openslide.lowlevel import _convert
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
## Must be openslide version 3.4.1
from openslide import open_slide, __library_version__ as openslide_version
import os
level = [3,4,5]

def load_image(slide_path, tumor_mask_path):

    # Loading the slide image and the tumor mask image
    slide = open_slide(slide_path)
    tumor_mask = open_slide(tumor_mask_path)

    # Checking if the dimensions of the mask image and the slide image match or not
    for i in range(len(slide.level_dimensions)-1):
        assert tumor_mask.level_dimensions[i][0] == slide.level_dimensions[i][0]
        assert tumor_mask.level_dimensions[i][1] == slide.level_dimensions[i][1]

    # Verify downsampling works as expected
    width, height = slide.level_dimensions[7]
    assert width * slide.level_downsamples[7] == slide.level_dimensions[0][0]
    assert height * slide.level_downsamples[7] == slide.level_dimensions[0][1]

    return slide, tumor_mask

def _load_image_lessthan_2_29(buf, size):
    '''buf must be a mutable buffer.'''
    _convert.argb2rgba(buf)
    return PIL.Image.frombuffer('RGBA', size, buf, 'raw', 'RGBA', 0, 1)

def _load_image_morethan_2_29(buf, size):
    '''buf must be a buffer.'''

    # Load entire buffer at once if possible
    MAX_PIXELS_PER_LOAD = (1 << 29) - 1
    # Otherwise, use chunks smaller than the maximum to reduce memory
    # requirements
    PIXELS_PER_LOAD = 1 << 26

    def do_load(buf, size):
        '''buf can be a string, but should be a ctypes buffer to avoid an
        extra copy in the caller.'''
        # First reorder the bytes in a pixel from native-endian aRGB to
        # big-endian RGBa to work around limitations in RGBa loader
        rawmode = (sys.byteorder == 'little') and 'BGRA' or 'ARGB'
        buf = PIL.Image.frombuffer('RGBA', size, buf, 'raw', rawmode, 0, 1)
        # Image.tobytes() is named tostring() in Pillow 1.x and PIL
        buf = (getattr(buf, 'tobytes', None) or buf.tostring)()
        # Now load the image as RGBA, undoing premultiplication
        return PIL.Image.frombuffer('RGBA', size, buf, 'raw', 'RGBa', 0, 1)

    # Fast path for small buffers
    w, h = size
    if w * h <= MAX_PIXELS_PER_LOAD:
        return do_load(buf, size)

    # Load in chunks to avoid OverflowError in PIL.Image.frombuffer()
    # https://github.com/python-pillow/Pillow/issues/1475
    if w > PIXELS_PER_LOAD:
        # We could support this, but it seems like overkill
        raise ValueError('Width %d is too large (maximum %d)' %
                         (w, PIXELS_PER_LOAD))
    rows_per_load = PIXELS_PER_LOAD // w
    img = PIL.Image.new('RGBA', (w, h))
    for y in range(0, h, rows_per_load):
        rows = min(h - y, rows_per_load)
        if sys.version[0] == '2':
            chunk = buffer(buf, 4 * y * w, 4 * rows * w)
        else:
            # PIL.Image.frombuffer() won't take a memoryview or
            # bytearray, so we can't avoid copying
            chunk = memoryview(buf)[y * w:(y + rows) * w].tobytes()
        img.paste(do_load(chunk, (w, rows)), (0, y))
    return img    

def read_slide(slide, x, y, level, width, height, as_float=False):

    # Reading the slides and converting them into a RGB numpy array
    openslide.lowlevel._load_image = _load_image_morethan_2_29
    im = slide.read_region((x, y), level, (width, height))
    im = im.convert('RGB')  # drop the alpha channel
    if as_float:
        im = np.asarray(im, dtype=np.float32)
    else:
        im = np.asarray(im)
    assert im.shape == (height, width, 3)
    return im

def find_tissue_pixels(image, intensity=0.8):

    # Finding the pixels having value less than or equal to the intensity value
    im_gray = rgb2gray(image)
    assert im_gray.shape == (image.shape[0], image.shape[1])
    indices = np.where(im_gray <= intensity)
    return zip(indices[0], indices[1])



def apply_mask(im, mask, color=1):

    # Applies the mask to the slides image
    masked = np.zeros((im.shape[0], im.shape[1]))
    for x, y in mask:
        masked[x][y] = color
    return masked    

def initialize_directories_test(slide_path, level):

    BASE_DIR = os.getcwd()

    img_num = slide_path.split('_')[1].strip(".tif")

    DATA = 'data/'
    IMG_NUM_FOLDER = img_num + '/'
    LEVEL_FOLDER = 'level_'+str(level)+'/'
    TISSUE_FOLDER = 'tissue_only/'
    ALL_FOLDER = 'all/'

    DATA_DIR = os.path.join(BASE_DIR, DATA)
    IMG_NUM_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER)
    LEVEL_NUM_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, LEVEL_FOLDER)
    TISSUE_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, LEVEL_FOLDER, TISSUE_FOLDER)
    ALL_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, LEVEL_FOLDER, ALL_FOLDER)

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(IMG_NUM_DIR):
        os.mkdir(IMG_NUM_DIR)
    if not os.path.exists(LEVEL_NUM_DIR):
        os.mkdir(LEVEL_NUM_DIR)
    if not os.path.exists(TISSUE_DIR):
        os.mkdir(TISSUE_DIR)
    if not os.path.exists(ALL_DIR):
        os.mkdir(ALL_DIR)

    return DATA + IMG_NUM_FOLDER + LEVEL_FOLDER + TISSUE_FOLDER, DATA + IMG_NUM_FOLDER + LEVEL_FOLDER + ALL_FOLDER



def split_image_test(im, tissue_mask, num_pixels, level_num, slide_path):
    x, y = im.shape[0], im.shape[1]
    x_count, y_count = int(np.ceil(x / num_pixels)), int(np.ceil(y / num_pixels))

    tissue_folder, all_folder = initialize_directories_test(slide_path, level_num)

    try:
        for i in range(x_count):
            for j in range(y_count):
                im_slice = np.zeros((num_pixels, num_pixels, 3))
                im_tissue_slice = np.zeros((num_pixels, num_pixels, 3))
                tissue_mask_slice = np.zeros((num_pixels, num_pixels))

                string_name = 'img_' + str(i * y_count + j)

                # Logic to handle the edges of the images
                if i == x_count - 1:
                    ub_x = x
                    assign_x = x - (x_count - 1) * num_pixels
                else:
                    ub_x = (i + 1) * num_pixels
                    assign_x = num_pixels

                if j == y_count - 1:
                    ub_y = y
                    assign_y = y - (y_count - 1) * num_pixels
                else:
                    ub_y = (j + 1) * num_pixels
                    assign_y = num_pixels

                tissue_mask_slice[0:assign_x, 0:assign_y] = tissue_mask[(i * num_pixels):ub_x, (j * num_pixels):ub_y]

                try:
                    if np.mean(tissue_mask_slice) > 0.7:
                        im_tissue_slice[0:assign_x, 0:assign_y, :] = im[(i * num_pixels):ub_x, (j * num_pixels):ub_y, :]
                        im_file_name_tissue = tissue_folder + string_name + ".jpg"
                        cv2.imwrite(im_file_name_tissue, im_tissue_slice)

                    im_slice[0:assign_x, 0:assign_y, :] = im[(i * num_pixels):ub_x, (j * num_pixels):ub_y, :]
                    im_file_name_all = all_folder + string_name + ".jpg"
                    cv2.imwrite(im_file_name_all, im_slice)

                except Exception as oerr:
                    print('Error with saving:', oerr)

    except Exception as oerr:
        print('Error with slicing:', oerr)    

def load_second_level(slide_path, input_level, num_input_pixels, output_level, num_output_pixels):
    img_num = slide_path.split('_')[1].strip(".tif")

    BASE_DIR = os.getcwd()
    DATA = 'data/'
    LEVEL_INPUT_FOLDER = 'level_' + str(input_level) + '/'
    LEVEL_OUTPUT_FOLDER = 'level_' + str(output_level) + '/'
    TISSUE_FOLDER = 'tissue_only/'
    ALL_FOLDER = 'all/'

    TISSUE_DIR_INPUT = os.path.join(BASE_DIR, DATA, img_num, LEVEL_INPUT_FOLDER, TISSUE_FOLDER)
    ALL_DIR_INPUT = os.path.join(BASE_DIR, DATA, img_num, LEVEL_INPUT_FOLDER, ALL_FOLDER)
    LEVEL_DIR_OUTPUT = os.path.join(BASE_DIR, DATA, img_num, LEVEL_OUTPUT_FOLDER)
    TISSUE_DIR_OUTPUT = os.path.join(BASE_DIR, DATA, img_num, LEVEL_OUTPUT_FOLDER, TISSUE_FOLDER)
    ALL_DIR_OUTPUT = os.path.join(BASE_DIR, DATA, img_num, LEVEL_OUTPUT_FOLDER, ALL_FOLDER)

    if not os.path.exists(LEVEL_DIR_OUTPUT):
        os.mkdir(LEVEL_DIR_OUTPUT)
    if not os.path.exists(TISSUE_DIR_OUTPUT):
        os.mkdir(TISSUE_DIR_OUTPUT)
    if not os.path.exists(ALL_DIR_OUTPUT):
        os.mkdir(ALL_DIR_OUTPUT)

    data_root_tissue_input = pathlib.Path(TISSUE_DIR_INPUT)
    all_image_paths_tissue_input = list(data_root_tissue_input.glob('*'))
    all_paths_tissue_str_input = [str(path) for path in all_image_paths_tissue_input]
    num_tissue_images_input = len(all_image_paths_tissue_input)

    data_root_all_input = pathlib.Path(ALL_DIR_INPUT)
    all_image_paths_all_input = list(data_root_all_input.glob('*'))
    all_paths_all_str_input = [str(path) for path in all_image_paths_all_input]
    num_all_images_input = len(all_paths_all_str_input)

    slide = open_slide(slide_path)
    input_width, input_height = slide.level_dimensions[input_level][0], slide.level_dimensions[input_level][1]
    output_width, output_height = slide.level_dimensions[output_level][0], slide.level_dimensions[output_level][1]
    slide = read_slide(slide,
                       x=0,
                       y=0,
                       level=output_level,
                       width=output_width,
                       height=output_height)

    # Find number of images that can fit in x and y direction of input given input number of pixels
    row_count, col_count = int(np.ceil(input_height / num_input_pixels)), int(np.ceil(input_width / num_input_pixels))        

def gen_image_paths_train(training_image_path_list, num_level):
    all_images_image_paths = []
    all_images_image_labels = []

    for i in training_image_path_list:

        slide_path = i

        img_num = slide_path.split('_')[1].strip(".tif")

        data_root_tumor = pathlib.Path('data/' + img_num + '/level_' + str(num_level) + '/tumor')
        all_image_paths_tumor = list(data_root_tumor.glob('*'))
        num_tumor_images = len(all_image_paths_tumor)

        data_root_notumor = pathlib.Path('data/' + img_num + '/level_' + str(num_level) + '/no_tumor')
        all_image_paths_notumor = list(data_root_notumor.glob('*'))
        random.shuffle(all_image_paths_notumor)
        all_image_paths_notumor = all_image_paths_notumor[0:num_tumor_images]

        all_image_paths = [str(path) for path in all_image_paths_tumor + all_image_paths_notumor]
        random.shuffle(all_image_paths)

        data_root = pathlib.Path('data/' + img_num + '/level_' + str(num_level))
        label_names = sorted(item.name for item in data_root.glob('*') if item.is_dir())
        label_to_index = dict((name, index) for index, name in enumerate(label_names))

        all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                            for path in all_image_paths]

        #update all image path lists
        all_images_image_paths = all_images_image_paths + all_image_paths
        all_images_image_labels = all_images_image_labels + all_image_labels

    return all_images_image_paths, all_images_image_labels

# Load and preprocess image.
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

#Preprocess image
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    image /= 255.0  # normalize to [0,1] range

    return image

# Generate image paths
def gen_image_paths(slide_path, level_num):
    img_num = slide_path.split('_')[1].strip(".tif")
    img_test_folder = 'tissue_only'

    data_root = pathlib.Path('data/' + img_num + '/level_' + str(level_num) +'/' + img_test_folder)

    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]

    return  all_image_paths

# Tumor predict mask.
def tumor_predict_mask(test, all_image_paths, depth, width):

    test = test[0:len(all_image_paths), :]
    img_num = np.zeros(len(all_image_paths))
    for i in range(len(all_image_paths)):
        img_num[i] = int(all_image_paths[i].strip('.jpg').split('/')[-1].split('_')[-1])

    # depth, width = int(np.ceil(slide_image.shape[0] / pixel_num)), int(np.ceil(slide_image.shape[1] / pixel_num))

    predictions = np.zeros((depth, width))
    conf_threshold = 0.85

    for i in range(len(test)):
        y = int(img_num[i] // width)
        x = int(np.mod(img_num[i], width))
        predictions[y, x] = int(test[i][1] > conf_threshold)

    return predictions

def create_tf_dataset_train(all_image_paths_1, all_image_paths_2, all_image_paths_3, all_image_labels):

    path_ds_1 = tf.data.Dataset.from_tensor_slices(all_image_paths_1)
    image_ds_1 = path_ds_1.map(load_and_preprocess_image, num_parallel_calls=8)

    path_ds_2 = tf.data.Dataset.from_tensor_slices(all_image_paths_2)
    image_ds_2 = path_ds_2.map(load_and_preprocess_image, num_parallel_calls=8)

    path_ds_3 = tf.data.Dataset.from_tensor_slices(all_image_paths_3)
    image_ds3_3 = path_ds_3.map(load_and_preprocess_image, num_parallel_calls=8)

    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

    image_label_ds = tf.data.Dataset.zip(((image_ds_1,image_ds_2, image_ds3_3), label_ds))

    BATCH_SIZE = 4

    steps_per_epoch = int(np.ceil(len(all_image_paths_1)/BATCH_SIZE))

    # Setting a shuffle buffer size larger than the dataset ensures that the data is completely shuffled.
    ds = image_label_ds.repeat()
    ds = ds.shuffle(buffer_size=4000)
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` lets the dataset fetches batches, asynchronously while the model is training.
    ds = ds.prefetch(1)


    return ds, steps_per_epoch

# Create tf dataset
def create_tf_dataset(all_image_paths_1, all_image_paths_2, all_image_paths_3):
    path_ds_1 = tf.data.Dataset.from_tensor_slices(all_image_paths_1)
    image_ds_1 = path_ds_1.map(load_and_preprocess_image, num_parallel_calls=8)

    path_ds_2 = tf.data.Dataset.from_tensor_slices(all_image_paths_2)
    image_ds_2 = path_ds_2.map(load_and_preprocess_image, num_parallel_calls=8)

    path_ds_3 = tf.data.Dataset.from_tensor_slices(all_image_paths_3)
    image_ds_3 = path_ds_3.map(load_and_preprocess_image, num_parallel_calls=8)
    image_test_ds = tf.data.Dataset.zip(((image_ds_1,image_ds_1, image_ds_1),))

    ## Dataset parameters
    BATCH_SIZE = 4

    steps_per_epoch = int(np.ceil(len(all_image_paths_1) / BATCH_SIZE))
    # Setting a shuffle buffer size larger than the dataset ensures that the data is completely shuffled.
    ds = image_test_ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` lets the dataset fetches batches, asynchronously while the model is training.
    ds = ds.prefetch(1)

    return ds, steps_per_epoch

# Test
def test_part(training_image_path, model, tissue_regions, slide_image_test,
                mask_image, depth, width, num_level_1, num_level_2, num_level_3):

    ## Generate image paths and labels
    all_image_paths_1 = gen_image_paths(training_image_path, num_level_1)

    # create the second file path to mimic the 1st
    all_image_paths_2 = []
    for i in all_image_paths_1:
        split_str = i.split('/')
        split_str[2] = 'level_'+str(num_level_2)
        path_2_string = ''
        for j in split_str:
            if j == split_str[-1]:
                path_2_string = path_2_string + j
            else:
                path_2_string = path_2_string + j + '/'
        all_image_paths_2.append(path_2_string)

    all_image_paths_3 = []
    for i in all_image_paths_1:
        split_str = i.split('/')
        split_str[2] = 'level_'+str(num_level_3)
        path_3_string = ''
        for j in split_str:
            if j == split_str[-1]:
                path_3_string = path_3_string + j
            else:
                path_3_string = path_3_string + j + '/'
        all_image_paths_3.append(path_3_string)


    ## Create tf.Dataset for testing
    ds_test, steps_per_epoch_test = create_tf_dataset(all_image_paths_1, all_image_paths_2, all_image_paths_3)

    ## Predict on test data
    test_predicts = model.predict(ds_test, steps = steps_per_epoch_test)

    ## Create mask containing test predictions
    predictions = tumor_predict_mask(test_predicts, all_image_paths_1, depth, width)

    plt.figure(figsize=(10,10))
    plt.title('Slide image')
    plt.axis('off')
    plt.imshow(slide_image_test,cmap='jet')

    plt.title('Predicted image')
    plt.axis('off')
    plt.imshow(slide_image_test)
    plt.imshow(predictions,cmap='jet', alpha=0.5)

    plt.title('Actual mask')
    plt.axis('off')
    plt.imshow(slide_image_test)
    plt.imshow(mask_image,cmap='jet', alpha=0.5)

    #heatmap_evaluation(predictions, mask_image, tissue_regions)


def train_part(training_image_path_list, num_level_1, num_level_2, num_level_3 ):

    # change input here from a specific image to an image path
    all_image_paths_1, all_image_labels_1 = gen_image_paths_train(training_image_path_list, num_level_1)

    # create the second file path to mimic the 1st
    all_image_paths_2 = []
    for i in all_image_paths_1:
        split_str = i.split('/')
        split_str[2] = 'level_'+str(num_level_2)
        path_2_string = ''
        for j in split_str:
            if j == split_str[-1]:
                path_2_string = path_2_string + j
            else:
                path_2_string = path_2_string + j + '/'
        all_image_paths_2.append(path_2_string)

    all_image_paths_3 = []
    for i in all_image_paths_1:
        split_str = i.split('/')
        split_str[2] = 'level_'+str(num_level_3)
        path_3_string = ''
        for j in split_str:
            if j == split_str[-1]:
                path_3_string = path_3_string + j
            else:
                path_3_string = path_3_string + j + '/'
        all_image_paths_3.append(path_3_string)

    ## Create tf.Dataset for training
    ds, steps_per_epoch = create_tf_dataset_train(all_image_paths_1, all_image_paths_2, all_image_paths_3, all_image_labels_1)

    return ds, steps_per_epoch


st.write("""
         # Metastasis Tumor Segmentation
         """
         )

#file = st.file_uploader("Please upload an image file", type=["jpg", "png", "tif"])

#if file is None:
 #   st.text("Please upload an image file")
#else:
 #   st.text("Image file uploaded successfully")


model = keras.models.load_model("/content/A/DenseNet 121_Metastasis/trained_weights_final")

if model is None:
    st.text("Trained Model is not uploaded")
else:
    st.text("Trained Model is uploaded successfully")
st.write(model) 

model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.sparse_categorical_crossentropy, metrics = ['acc'])   
user_input = st.text_input("Give link of WSI image in TIF format")
if user_input is None:
  st.text("Please provide link")
else:  
  testing_image_path_list = []
  testing_image_path_list.append(user_input)

  slide_path_test = testing_image_path_list[0]
  tumor_mask_path_test = slide_path_test.split('.')[0]+'_mask.tif'
  st.text(str(slide_path_test))                                                                                                 
  st.text(str(tumor_mask_path_test))   

  slide, tumor_mask = load_image(slide_path_test, tumor_mask_path_test) 
  st.text("Slide extracted")  
  st.text("Tumor mask created") 
  width, height = slide.level_dimensions[3][0], slide.level_dimensions[3][1]
  st.text("width acquired")  
  st.text("height acquired") 



  slide = read_slide(slide,x=0,y=0,level=3,width=width,height=height)

  st.text("Slide is read")

  tumor_mask = read_slide(tumor_mask,x=0,y=0,level=3,width=width,height=height)

  st.text("Tumor mask is read")

  image_depth, image_width = int(np.ceil(slide.shape[0] / 64)), int(np.ceil(slide.shape[1] / 64))
  st.text("image_depth is made")
  st.text("image_width  is made")

  tumor_mask = tumor_mask[:, :, 0]
  st.text("Tumor mask Shaped properly")



  tissue_pixels = list(find_tissue_pixels(slide))

  

  tissue_regions = apply_mask(slide, tissue_pixels)   

  

  split_image_test(slide, tissue_regions, 64, 3, testing_image_path_list[0])

  st.text("Slide Image test")



  load_second_level(testing_image_path_list[0],input_level = 3,num_input_pixels = 64,output_level = 4,num_output_pixels = 64)

  st.text("Second level patches obtained")

  load_second_level(testing_image_path_list[0],input_level = 3,num_input_pixels = 64,output_level = 5,num_output_pixels = 64)

  st.text("Third level patches obtained")





  all_image_paths_1 = gen_image_paths(testing_image_path_list[0],3)

  st.text("Image path 1 created")
  all_image_paths_2 = []
  for i in all_image_paths_1:
      split_str = i.split('/')
      split_str[2] = 'level_'+str(level[1])
      path_2_string = ''
      for j in split_str:
          if j == split_str[-1]:
              path_2_string = path_2_string + j
          else:
              path_2_string = path_2_string + j + '/'
      all_image_paths_2.append(path_2_string)

  st.text("Image path 2 created")

  all_image_paths_3 = []
  for i in all_image_paths_1:
      split_str = i.split('/')
      split_str[2] = 'level_'+str(level[2])
      path_3_string = ''
      for j in split_str:
          if j == split_str[-1]:
              path_3_string = path_3_string + j
          else:
              path_3_string = path_3_string + j + '/'
      all_image_paths_3.append(path_3_string)  

  st.text("Image path 3 created") 


  ds_test, steps_per_epoch_test = create_tf_dataset(all_image_paths_1, all_image_paths_2, all_image_paths_3)

  st.text("Test Dataset created")
  st.text("Prediction started")
  test_predicts = model.predict(ds_test, steps = steps_per_epoch_test)
  st.text("Prediction ended")
  st.text("Prediction done")
  predictions = tumor_predict_mask(test_predicts, all_image_paths_1, image_depth, image_width)

  A=plt.figure(figsize=(50,20))
  plt.subplot(1,3,1)
  plt.imshow(slide)
  plt.imshow(tumor_mask,cmap='copper', alpha=0.5)
  plt.grid(False)
  plt.subplot(1,3,2)
  plt.imshow(slide)
  plt.imshow(predictions, cmap='copper') 
  plt.grid(False)
  plt.subplot(1,3,3)
  plt.imshow(tumor_mask, cmap='copper') 
  plt.grid(False)
  st.pyplot(A)