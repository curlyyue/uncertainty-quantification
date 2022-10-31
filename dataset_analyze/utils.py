import os
import json
from sklearn.feature_extraction import img_to_graph
from PIL import Image, ImageDraw, ImageColor, ImageFont
import matplotlib.pyplot as plt



def split_annos(TypeKeys:list, Path:str):
    """
    to split annoations according to type of dataset
    TypeKeys: image keys of dataset, like train_keys
    Path: annotation path
    """
    files = os.listdir(Path)
    annos = [] # stores the annotation
    fnames = [] # name of corresponding picture

    for file in files:
        # if not os.path.isdir(file):
        if file[:-5] in TypeKeys:
            f = open(Path+'/'+file)
            fnames.append(file[:-5])
            annos.append(json.load(f))    
    # annotation = list(zip(fnames, annos))
    return fnames, annos
