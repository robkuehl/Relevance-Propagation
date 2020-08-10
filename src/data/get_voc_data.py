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
from sklearn.utils import shuffle

class pascal_data_generator():
    """
    Klasse zum erzeugen von Datensätzen aus dem VOC Devkit
    - init params:
        :voc_path (Path): Pfad zum VOC Devkit aus dem die xml und png files geladen werden 
        :desired_size (int): geüwnschte Größe für die Bilder im Trainingsdatensatz. Default: 224 für VGG16
        :override (bool): Falls True werden die bereits existierende gepaddete Bilder neu erstellt und im Speicher überschrieben
    - Methoden:
        :zero_pad_images: reshapen der Bilder mittels Zeropadding auf quadratische Größe mit Seitenlänge desired_size
        :reshape_images: reshapen der Bilder mittels Stauchen und Strecken auf quadratische Größe mit Seitenlänge desired_size
        :images_to_numpy: konvertieren der Bilddateien in Numby arrays und abspeichern als pickle Dateien
        :get_voc_labels: Erzeugt Dataframe mit allen Bildern und entsprechenden label. Lädt Label Dataframe für Teilmenge von Klassen.
        :get_voc_images: Laden der VOC Bilder
        :get_training_data: Erzeugen eines Trainigsdatensatzen
        
    - Methodenbaum:
        User: ruft get_training_data auf
        get_training_data: ruft get_voc_labels und get_voc_images auf
        get_voc_images: ruft images_to_numpy auf zero_pad_images und reshape_images auf
    """   

    def __init__(self, desired_size:int=224, override=False):
        dirname = os.path.dirname(__file__)
        self.voc_path = os.path.join(dirname, '../../../VOCdevkit/VOC2012')
        self.desired_size = desired_size
        self.override=override



    """
    Methode zum reshapen der Bilder des Pascal Voc Datensatzes mittels Zeropadding, d.h. Hinzufügen eines schwarzen Randes.
    Das Bild wird geladen und mit Hilfe der Python Image Library (PIL) auf entsprechenende Größe formatiert.
    """
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
 
               
                

    """
    Methode zum reshapen der Bilder des Pascal Voc Datensatzes mittels Stauchen und Strecken.
    Das Bild wird geladen und mit Hilfe der Python Image Library (PIL) auf entsprechenende Größe formatiert.
    """
    def reshape_images(self):
        if not os.path.isdir(pathjoin(self.voc_path, 'JPEGImages_reshaped')):
            os.mkdir(pathjoin(self.voc_path, 'JPEGImages_reshaped'))
        
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
    """
    Methode zum konvenrtieren von Bilddateien in Numpy Arrays.
    Für den angegeben image_type werden die Bilder aus dem entsprechenden Ordner geladen, in Numpy konvertiert und als pickle Datei gespeichert.
    - params:
        :image_type (str): Bilder die wir in Numpy konvertieren wollen
    """
    def images_to_numpy(self, image_type):
        if image_type == 'regular':
            source_path = pathjoin(self.voc_path, 'JPEGImages')
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
            if not os.path.isfile(pathjoin(store_path, filename+'.pickle')):
                image_path = pathjoin(source_path, filename+'.jpg')
                image = np.asarray(Image.open(image_path))
                with open(pathjoin(store_path, filename+'.pickle'), 'wb') as file:
                    pickle.dump(image, file)






    """
    Methode zum erstellen des Labels Dataframe für eine vorgegebene Liste von Klassen.
    Falls es noch keinen Datframe für alle Bilder und Label gibt, wird dieser erzeugt und gespeichert.
    Im Sinne der Regularisierung werden Bilder hinzugefügt, auf denen keines der gwünschten Label zu sehen ist.
    Um die größe der Klassen zu reduzieren, werden pro Bildklasse per Zufall so viele Bilder gezogen, wie sie in der Klasse vom geringster Größe enthalten sind.
    Da es sich um eine Multilabel Klassifizierung handelt, werden die Klassen jedoch nicht notwendigerweise gleich große sein.
    - params:
        :classes (list): Liste von Klassen deren Bilder verwendet werden sollen 
    """

    def get_voc_labels(self, classes):
        # falls es bereits eine Datei gibt, in der der Label Dataframe für alle Bilder gespeichert ist, dann lade den DF
        if os.path.isfile(pathjoin(self.voc_path, 'label_df.pickle')):
            label_df = pd.read_pickle(pathjoin(self.voc_path, 'label_df.pickle'))
        # Andernfalls, erzeuge mittels der XML Dateien diesen Dataframe und speichere ihn als pickle Datei
        else:
            xml_path = pathjoin(self.voc_path, 'Annotations')
            image_names = []
            labels = []
            for file in os.listdir(pathjoin(self.voc_path, 'JPEGImages')):
                if 'jpg' not in file:
                    continue
                tree = ElementTree.parse(pathjoin(xml_path, os.path.splitext(file)[0]+'.xml'))
                root = tree.getroot()
                label = set([obj.find('name').text for obj in root.findall('.//object')])
                labels.append(label)
                image_names.append(os.path.splitext(file)[0])

            mlb = MultiLabelBinarizer()

            label_df = pd.DataFrame(mlb.fit_transform(labels),
                            columns=mlb.classes_,
                            index=image_names)
            
            
            label_df.to_pickle(pathjoin(self.voc_path, 'label_df.pickle'))
            
        # Laden den Dataframe für die angegebene Menge von Klassen
        if len(classes)!=0:
            # choose only columns of the given classes
            # choose only rows where at least one label is 1
            new_label_df = label_df[(label_df[classes]!=np.zeros(len(classes))).any(axis=1)][classes]
            
            # ergänze mit zufälliger Auswahl von Bildern auf denen keines der Label enthalten ist
            rand_select_df = label_df[(label_df[classes]==np.zeros(len(classes))).all(axis=1)][classes]
            try:
                new_label_df = new_label_df.append(rand_select_df.sample(int(new_label_df.shape[0]*0.25),
                                                                         random_state=42))
            except ValueError:
                # Falls im random_select_df weniger als 25% der Menge von Bildern sind wie im new_label_df
                new_label_df = label_df
        else:
            new_label_df = label_df
                
        # reduce labels to more equal size for all classes
        reduced_label_df = pd.DataFrame(columns=new_label_df.columns, dtype=np.float64)
        cl_size = min(new_label_df.sum())
        print('\nSize of classes:\n{}'.format(new_label_df.sum()))
        for cl in list(new_label_df.columns):
            sample_df = new_label_df[new_label_df[cl]==1].sample(cl_size, random_state=42)
            reduced_label_df = pd.concat([reduced_label_df, sample_df])
            
        new_label_df = reduced_label_df
        
        
        print('\nKlassen im Datensatz:\n{}'.format(new_label_df.sum()))
            
        return new_label_df




    """
    Methode zum laden der Bilder im Numpy Format aus den Pickle Dateien
    - params:
        :image_type (str): Bildart die wir zum Training verwenden wollen
        :label_df (Dataframe): Dataframe mit Bildnamen im Index und One-Hot-Encoded Labeln (mit get_voc_labels erzeugt)
            
    """
    def get_voc_images(self, image_type:str, label_df:pd.DataFrame):
        # die Dateinamen der zu ladenden Bilder sind im Index des Label Dataframe gespeichert
        image_names = list(label_df.index)
        
        if image_type == 'regular':
            source_path = pathjoin(self.voc_path, 'JPEGImages', 'Images_as_pickle')
        elif image_type == 'zero_padded':
            source_path = pathjoin(self.voc_path, 'JPEGImages_zero_padded', 'Images_as_pickle')
        elif image_type == 'reshaped':
            source_path = pathjoin(self.voc_path, 'JPEGImages_reshaped', 'Images_as_pickle')
        
        self.images_to_numpy(image_type)
        
        # Lade die Bilder aus den Pickle Dateien
        images = []
        for file in image_names:
            image_path = pathjoin(source_path, file+'.pickle')
            with open(image_path, 'rb') as image_file:
                images.append(pickle.load(image_file))
                
        return np.asarray(images)



    """
    Methode zu Laden eines Trainingsdatensatzs für Pascal VOC
    - params:
        :classes (list): Liste von Klassen die im Trainingsdatensatz enthalten sein sollen
        :dataset (string): Name des Datensatzes der verwendet werden soll
            1) pascal_voc_regular
            2) pascal_voc_zero_padded
            3) pascal_voc_reshaped
            
    """
    def get_training_data(self, classes, dataset):

        # lade die Label aller Bilder die die angegebenen Klassen enthalten
        label_df = self.get_voc_labels(classes=classes)
        classes = list(label_df.columns)
        
        # lade die Bilder deren Dateiname im Label Dataframe gespeichert sind als Numpy für den angegebenen Datensatz
        images = self.get_voc_images(image_type=dataset[11:], label_df=label_df)
        
        # Erzeuge Trainings- und Testdaten
        train_images, test_images, train_labels_df, test_labels_df = train_test_split(images, label_df, test_size=0.1, random_state=42)
        
        # Speichere die erzeugte Daten in einem Dictionary
        data = {'train_images': train_images,
                    'train_labels_df': train_labels_df,
                    'test_images': test_images,
                    'test_labels_df': test_labels_df
            }
        
        return data, dict(zip([i for i in range(len(classes))], classes))