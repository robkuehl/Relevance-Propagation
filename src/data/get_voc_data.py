import os
from pathlib import Path
from xml.etree import ElementTree
from os.path import join as pathjoin
import numpy as np
from PIL import Image
import gc
import pickle
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import time
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps

# voc_path = '/home/robin/Documents/VOCdevkit/VOC2012'


# Method to load images and store them as numpy binarys

def zero_pad_images(voc_path, desired_size=250):
    if not os.path.isdir(pathjoin(voc_path, 'JPEGImages_zero_padded')):
        os.mkdir(pathjoin(voc_path, 'JPEGImages_zero_padded'))
    
    for file in os.listdir(pathjoin(voc_path, 'JPEGImages')):
        if 'jpg' not in file:
            continue
        im_pth = pathjoin(voc_path, 'JPEGImages', file)


        im = Image.open(im_pth)
        old_size = im.size  # old_size[0] is in (width, height) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        # use thumbnail() or resize() method to resize the input image

        # thumbnail is a in-place operation

        #im.thumbnail(new_size, Image.BICUBIC)

        im = im.resize(new_size, Image.BICUBIC)
        # create a new image and paste the resized on it

        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im, ((desired_size-new_size[0])//2,
                            (desired_size-new_size[1])//2))
        
        new_im.save(pathjoin(voc_path, 'JPEGImages_zero_padded', file))
            


def reshape_images(voc_path, desired_size=250):
    if not os.path.isdir(pathjoin(voc_path, 'JPEGImages_reshaped')):
        os.mkdir(pathjoin(voc_path, 'JPEGImages_reshaped'))
    else:
        pass
    
    for file in os.listdir(pathjoin(voc_path, 'JPEGImages')):
        if os.path.isfile(pathjoin(voc_path, 'JPEGImages_reshaped', file)):
            continue
        if 'jpg' not in file:
            continue
        
        im_pth = pathjoin(voc_path, 'JPEGImages', file)

        im = Image.open(im_pth)
        im = im.resize((desired_size, desired_size), Image.BICUBIC)
        
        im.save(pathjoin(voc_path, 'JPEGImages_reshaped', file))
        
    


def images_to_numpy(voc_path, image_type):
    if image_type == 'regular':
        source_path = pathjoin(voc_path, 'JPEGImages')
    elif image_type == 'same_padded':
        source_path = pathjoin(voc_path, 'JPEGImages_same_padded')
    elif image_type == 'zero_padded':
        source_path = pathjoin(voc_path, 'JPEGImages_zero_padded')
    elif image_type == 'reshaped':
        source_path = pathjoin(voc_path, 'JPEGImages_reshaped')
    
    store_path = pathjoin(source_path,'Images_as_pickle')

    if not os.path.isdir(store_path):
        os.mkdir(store_path)

    for file in os.listdir(source_path):
        if 'jpg' not in file:
            continue
        filename = os.path.splitext(file)[0]
        if os.path.isfile(pathjoin(store_path, filename+'.pickle')):
            pass
        else:
            image_path = pathjoin(source_path, filename+'.jpg')
            image = np.asarray(Image.open(image_path))
            with open(pathjoin(store_path, filename+'.pickle'), 'wb') as file:
                pickle.dump(image, file)


# Create the labels to the corresponding images from the xml files and store the labels as pickle as well
                
def labels_to_pickle(voc_path):
    if os.path.isfile(pathjoin(voc_path, 'labels.pickle')):
        pass
    else:
        xml_path = pathjoin(voc_path, 'Annotations')
        labels = []
        for file in os.listdir(pathjoin(voc_path, 'JPEGImages')):
            if 'jpg' not in file:
                continue
            tree = ElementTree.parse(pathjoin(xml_path, os.path.splitext(file)[0]+'.xml'))
            root = tree.getroot()
            labels.append(set([obj.find('name').text for obj in root.findall('.//object')]))
        with open(pathjoin(voc_path, 'labels.pickle'), 'wb') as f:
            pickle.dump(labels, f)
            
            
            
# create a dataframe with binary labels

def get_voc_labels(voc_path):
    labels = []
    image_names = []
    for file in os.listdir(pathjoin(voc_path, 'JPEGImages')):
        if 'jpg' not in file:
            continue
        image_names.append(os.path.splitext(file)[0])
            
    with open(pathjoin(voc_path, 'labels.pickle'), 'rb') as label_file:
        labels = pickle.load(label_file)

    #label_series = pd.Series(labels)

    mlb = MultiLabelBinarizer()

    label_df = pd.DataFrame(mlb.fit_transform(labels),
                       columns=mlb.classes_,
                       index=image_names)
    
    return label_df



# Load the numpy-images from the pickle files

def get_voc_images(voc_path:Path, image_type:str, image_names:list):
    if image_type == 'regular':
        source_path = pathjoin(voc_path, 'JPEGImages', 'Images_as_pickle')
    elif image_type == 'same_padded':
        source_path = pathjoin(voc_path, 'JPEGImages_same_padded', 'Images_as_pickle')
    elif image_type == 'zero_padded':
        source_path = pathjoin(voc_path, 'JPEGImages_zero_padded', 'Images_as_pickle')
    elif image_type == 'reshaped':
        source_path = pathjoin(voc_path, 'JPEGImages_reshaped', 'Images_as_pickle')
    images = []
    for file in image_names:
        image_path = pathjoin(source_path, file+'.pickle')
        with open(image_path, 'rb') as image_file:
            images.append(pickle.load(image_file))
            
    return images


# method to load data for a specific set of labels
# it returns images for whom at least one of the labels is 1 and the corresponding label dataframe
# if classes is None or empty it returns the whole dataset

def get_voc_data(voc_path: Path, image_type:str, classes: list, training:bool):
    images_to_numpy(voc_path, image_type)
    labels_to_pickle(voc_path)
    label_df = get_voc_labels(voc_path)
    if classes == None or len(classes)==0:
        classes = list(label_df.columns)
    new_label_df = label_df[(label_df[classes]!=np.zeros(len(classes))).any(axis=1)][classes]
    images = get_voc_images(voc_path, image_type, list(new_label_df.index))
    
    
    return new_label_df, images