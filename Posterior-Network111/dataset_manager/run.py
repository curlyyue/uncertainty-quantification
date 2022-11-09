import os
import pandas as pd
from ClassificationDatasetTemp import CustomImageDataset

# load dataset
path = '/lab/project-1'
train_label = 'train_label'
label_path = os.path.join(path, 'train_label.csv')
img_dir = os.path.join(path,'train_signs')

transform = None

def split_dataset(anno_path:str, ind, ood):
    '''
    to split dataset according to list of in-distribution and ood
    '''
    label = pd.read_csv(anno_path)
    ind_list = []
    ood_list = []
    for i in range(len(label)):
        if label.iloc[i][4] in ind_list:
            ind_list.append(label.iloc[i])
        elif label.iloc[i][4] in ood_list:
            ood_list.append(label.iloc[i])
    return ind_list, ood_list


ood = ['g1', 'g2']
ind= ['g3', 'g4', 'g5', 'g6']

ind_labels, ood_labels = split_dataset(label_path, ind, ood)

ood_dataset = CustomImageDataset(ood_labels, img_dir, transform)
ind_dataset = CustomImageDataset(ind_labels, img_dir, transform)


# im, l = ood_dataset[1]
# print(l)