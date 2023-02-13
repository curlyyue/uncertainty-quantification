import os
import json
# from sklearn.feature_extraction import img_to_graph
from PIL import Image, ImageDraw, ImageColor, ImageFont
# import matplotlib.pyplot as plt



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


def split_g(label:str):
    '''
    Note: this is only for signs infor, warn, comple, regu
    to split signs with same g value
    we have g1-12, g15, g25, g45 in total 15
    
    '''
    num = 99
    if label[-2] == 'g':
        num = int(label[-1])-1
    elif (label[-2:]<='12'):
        num = int(label[-2:])-1
    elif (label[-2:]>'12'):
        if label[-2:]=='15':
            num = 12
        elif label[-2:] == '25':
            num = 13
        elif label[-2:] == '45':
            num = 14
        else:
            print('error', label)
    else:
        print('error', label)
    return num
