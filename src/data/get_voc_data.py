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

class pascal_data_generator():

    def __init__(self, desired_size:int=224, override=False):
        dirname = os.path.dirname(__file__)
        self.voc_path = os.path.join(dirname, '../../data/VOCdevkit/VOC2012')
        self.desired_size = desired_size
        self.override=override

    def zero_pad_images(self):
        if not os.path.isdir(pathjoin(self.voc_path, 'JPEGImages_zero_padded')):
            os.mkdir(pathjoin(self.voc_path, 'JPEGImages_zero_padded'))
        
        for file in os.listdir(pathjoin(self.voc_path, 'JPEGImages')):
            if os.path.isfile(pathjoin(self.voc_path, 'JPEGImages_zero_padded', file)) and self.override==False:
                continue
            if 'jpg' not in file:
                continue
            
            im_pth = pathjoin(self.voc_path, 'JPEGImages', file)


            im = Image.open(im_pth)
            old_size = im.size  # old_size[0] is in (width, height) format

            ratio = float(self.desired_size)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])
            # use thumbnail() or resize() method to resize the input image

            # thumbnail is a in-place operation

            #im.thumbnail(new_size, Image.BICUBIC)

            im = im.resize(new_size, Image.BICUBIC)
            # create a new image and paste the resized on it

            new_im = Image.new("RGB", (self.desired_size, self.desired_size))
            new_im.paste(im, ((self.desired_size-new_size[0])//2,
                                (self.desired_size-new_size[1])//2))
            
            new_im.save(pathjoin(self.voc_path, 'JPEGImages_zero_padded', file))
                


    def reshape_images(self):
        if not os.path.isdir(pathjoin(self.voc_path, 'JPEGImages_reshaped')):
            os.mkdir(pathjoin(self.voc_path, 'JPEGImages_reshaped'))
        else:
            pass
        
        for file in os.listdir(pathjoin(self.voc_path, 'JPEGImages')):
            if os.path.isfile(pathjoin(self.voc_path, 'JPEGImages_reshaped', file)) and self.override==False:
                continue
            if 'jpg' not in file:
                continue
            
            im_pth = pathjoin(self.voc_path, 'JPEGImages', file)

            im = Image.open(im_pth)
            im = im.resize((self.desired_size, self.desired_size), Image.BICUBIC)
            
            im.save(pathjoin(self.voc_path, 'JPEGImages_reshaped', file))
            
        

    # Method to load images and store them as numpy binarys
    def images_to_numpy(self, image_type):
        if image_type == 'regular':
            source_path = pathjoin(self.voc_path, 'JPEGImages')
        elif image_type == 'same_padded':
            source_path = pathjoin(self.voc_path, 'JPEGImages_same_padded')
        elif image_type == 'zero_padded':
            self.zero_pad_images()
            source_path = pathjoin(self.voc_path, 'JPEGImages_zero_padded')
        elif image_type == 'reshaped':
            self.reshape_images()
            source_path = pathjoin(self.voc_path, 'JPEGImages_reshaped')
        
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
                    
    def labels_to_pickle(self):
        if os.path.isfile(pathjoin(self.voc_path, 'labels.pickle')):
            pass
        else:
            xml_path = pathjoin(self.voc_path, 'Annotations')
            labels = []
            for file in os.listdir(pathjoin(self.voc_path, 'JPEGImages')):
                if 'jpg' not in file:
                    continue
                tree = ElementTree.parse(pathjoin(xml_path, os.path.splitext(file)[0]+'.xml'))
                root = tree.getroot()
                labels.append(set([obj.find('name').text for obj in root.findall('.//object')]))
            with open(pathjoin(self.voc_path, 'labels.pickle'), 'wb') as f:
                pickle.dump(labels, f)
                
                
                
    # create a dataframe with binary labels

    def get_voc_labels(self, classes):
        self.labels_to_pickle()
        labels = []
        image_names = []
        for file in os.listdir(pathjoin(self.voc_path, 'JPEGImages')):
            if 'jpg' not in file:
                continue
            image_names.append(os.path.splitext(file)[0])
                
        with open(pathjoin(self.voc_path, 'labels.pickle'), 'rb') as label_file:
            labels = pickle.load(label_file)

        #label_series = pd.Series(labels)

        mlb = MultiLabelBinarizer()

        label_df = pd.DataFrame(mlb.fit_transform(labels),
                        columns=mlb.classes_,
                        index=image_names)
        
        if classes == None or len(classes)==0:
            classes = list(label_df.columns)
        label_df = label_df[(label_df[classes]!=np.zeros(len(classes))).any(axis=1)][classes]
        
        return label_df



    # Load the numpy-images from the pickle files

    def get_voc_images(self, image_type:str, label_df:pd.DataFrame):
        image_names = list(label_df.index)
        if image_type == 'regular':
            source_path = pathjoin(self.voc_path, 'JPEGImages', 'Images_as_pickle')
        elif image_type == 'same_padded':
            source_path = pathjoin(self.voc_path, 'JPEGImages_same_padded', 'Images_as_pickle')
        elif image_type == 'zero_padded':
            source_path = pathjoin(self.voc_path, 'JPEGImages_zero_padded', 'Images_as_pickle')
        elif image_type == 'reshaped':
            source_path = pathjoin(self.voc_path, 'JPEGImages_reshaped', 'Images_as_pickle')
        self.images_to_numpy(image_type)
        images = []
        for file in image_names:
            image_path = pathjoin(source_path, file+'.pickle')
            with open(image_path, 'rb') as image_file:
                images.append(pickle.load(image_file))
                
        return np.asarray(images)


    def get_training_data(self, classes, dataset):
        label_df = self.get_voc_labels(classes=classes)
        if classes == ['person', 'horse']:
            horse_and_person_df = label_df[(label_df==[1,1]).all(axis=1)]
            horse_df = label_df[(label_df==[0,1]).all(axis=1)]
            person_df = label_df[(label_df==[1,0]).all(axis=1)]
            
            new_person_df = person_df.sample(horse_df.shape[0])
            label_df = pd.concat([horse_df, new_person_df, horse_and_person_df])
            
        images = self.get_voc_images(image_type=dataset[11:], label_df=label_df)
        
        train_images, test_images, train_labels_df, test_labels_df = train_test_split(images, label_df, test_size=0.2, random_state=42)
        
        data = {'train_images': train_images,
                    'train_labels_df': train_labels_df,
                    'test_images': test_images,
                    'test_labels_df': test_labels_df
            }
        
        return data, dict(zip([i for i in range(len(classes))], classes))