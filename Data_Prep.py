import glob
import numpy as np
import random
import fnmatch
import os
from PIL import Image
from matplotlib import pyplot as plt
import PIL
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
#from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

from bs4 import BeautifulSoup
from medpy.io import load
import cv2

from collections import namedtuple, deque
import math
from sklearn.utils import shuffle
from tqdm import tqdm

''' Read files for AD and CN
'''

def mask_load(path):
    files_id = []
    mask = []
    unique_id = []
    for root, folder, file in os.walk(os.path.abspath(path)):
        for masks in fnmatch.filter(file,"*.nii"):
            y = os.path.join(root, masks).rsplit("\\")[6]
            uid = os.path.join(root, masks).rsplit("\\")[8]
            mask.append(os.path.join(root, masks))
            files_id.append(y)
            unique_id.append(uid)
    return files_id, mask, unique_id

path_ad = "Path to AD\\ADNI"
all_files_ad_id, mask_ad, ad_unique_id = mask_load(path_ad)

path_cn = "Path to \\CN\\ADNI"
all_files_cn_id, mask_cn, cn_unique_id = mask_load(path_cn)


#MP-RAGE_, SAG_MP-RAGE, MP-RAGE-, MPRAGE, MPRAGE_, MP-RAGE

def data_load(path, file_id, unique_id):
  file_path = []
  f_path = []
  for root, folder, file in os.walk(os.path.abspath(path)):
    f_path = []
    for filename in fnmatch.filter(file, "*.dcm" ):
      id1 = os.path.join(root, filename).rsplit("\\")[6]
      if id1 in file_id:
          id2 = os.path.join(root, filename).rsplit("\\")[8]
          if id2 in unique_id:
              x = os.path.join(root, filename).rsplit("\\")[7]
              if (x == "MP-RAGE_" or x == "SAG_MP-RAGE" or x == "MP-RAGE-" or
                   x == "MPRAGE" or x == "MPRAGE_" or x == "MP-RAGE") :
                       f_path.append(os.path.join(root, filename))

    if len(f_path) != 0:
        file_path.append(f_path)
  return file_path

all_files_ad = data_load(path_ad, all_files_ad_id, ad_unique_id )

def data_load_cn(path, file_id, unique_id):
  file_path = []
  f_path = []
  for root, folder, file in os.walk(os.path.abspath(path)):
    f_path = []
    for filename in fnmatch.filter(file, "*.dcm" ):
      id1 = os.path.join(root, filename).rsplit("\\")[6]
      if id1 in file_id:
          id2 = os.path.join(root, filename).rsplit("\\")[8]
          if id2 in unique_id:
              x = os.path.join(root, filename).rsplit("\\")[7]
              if ( x == "MP-RAGE_" or x == "MP-RAGE-" or x == "MPRAGE" or
                   x == "SAG_MP-RAGE_" or x == "MPRAGE_" or x == "MP_RAGE" or
                   x == "MP-RAGE" or x == "MP-RAGE_" or x == "MP-RAGE__SERIES_2_" or
                   x == "SAG_MP-RAGE") :
                       f_path.append(os.path.join(root, filename))

    if len(f_path) != 0:
        file_path.append(f_path)
  return file_path

all_files_cn = data_load_cn(path_cn, all_files_cn_id, cn_unique_id)

''' Choose slices 25 - 125
'''
var1 = 15
var2 = 159
def slice_select(var1, var2, file_path):
    new_filepath = []
    for i in range(len(file_path)):
        dummy_var = []
        for j in range(len(file_path[i])):
            if j>=var1 and j<= var2:
                dummy_var.append(file_path[i][j])
        new_filepath.append(dummy_var)
    return new_filepath

sliced_all_files_cn = slice_select(var1, var2, all_files_cn)
sliced_all_files_ad = slice_select(var1, var2, all_files_ad)

''' mask selection
'''

def compute_all_masks(var1, var2, file):
    all_masks = []
    for i in range(len(file)):
        mask1,_ = load(file[i])
        dummy_var = []
        for j in range(mask1.shape[0]):
            if j >= var1 and j <= var2:
                msk = 255*mask1[j]/mask1[j].max()
                msk = cv2.resize(msk, (256,256))
                _, bw_img = cv2.threshold(msk.astype(np.uint8), 10, 255, cv2.THRESH_OTSU)
                # _, bw_img = cv2.threshold(masks[indx_mx], 127, 255, cv2.THRESH_BINARY)
                contours,_ = cv2.findContours(bw_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) != 0:
                    rect =  cv2.boundingRect(contours[0])
                    dummy_var.append([j, rect])
                else:
                    dummy_rect = [0,0,0,0]
                    dummy_var.append([j, dummy_rect])

        all_masks.append(dummy_var)
    return all_masks


sliced_masks_cn = compute_all_masks(var1, var2, mask_cn)
sliced_masks_ad = compute_all_masks(var1, var2, mask_ad)

''' Select mask dimension and markers for Hippocampus region
'''
def markers(slices_mask):
    marks = []
    for i in range(len(slices_mask)):
        x_min = []
        y_min = []
        w_max = []
        h_max = []
        z_mn = 0
        indx_diff = []
        for j in range(len(slices_mask[i])):
            x_min.append(slices_mask[i][j][1][0])
            y_min.append(slices_mask[i][j][1][1])
            w_max.append(slices_mask[i][j][1][2])
            h_max.append(slices_mask[i][j][1][3])

            if slices_mask[i][j][1][0] != 0:
                indx_diff.append(slices_mask[i][j][0])
        x_min = np.array(x_min)
        y_min = np.array(y_min)
        x_mn = np.min(x_min[np.nonzero(x_min)])
        y_mn = np.min(y_min[np.nonzero(y_min)])
        w_mx = np.array(w_max).max()
        h_mx = np.array(h_max).max()
        dim = [x_mn, y_mn, w_mx, h_mx]
        # print(dim)
        d_mx = indx_diff[-1] - indx_diff[0]
        # print(diff)
        z_mn = indx_diff[0]
        marks.append([x_mn, y_mn, z_mn, w_mx, h_mx , d_mx])
    return marks

sliced_markers_cn = markers(sliced_masks_cn)
sliced_markers_ad = markers(sliced_masks_ad)

''' Concat labels 1 - AD,  0 -  CN
'''
labels_ad = np.ones((len(sliced_markers_ad),1))
labels_cn = np.zeros((len(sliced_markers_cn),1))

label_cn = np.hstack([sliced_markers_cn, labels_cn])
label_ad = np.hstack([sliced_markers_ad, labels_ad])


class data_cleaning_concat():
    def __init__(self, data1, data2, label1, label2):
        self.data1 = data1
        self.data2 = data2
        self.label1 = label1
        self.label2 = label2

    def _index_values(self, dataset):
        index = []
        for i in range(len(dataset)):
            if len(dataset[i]) < 145:
                index.append(i)
        return index

    def _pop_list(self, dataset, label, index):
        for i in range(len(index)):
            dataset.pop(index[i])
            label.pop(index[i])
        return dataset, label

    def files_concat(self):
        concat_data = []
        concat_label = []

        data1_ , label1_ = self._pop_list(self.data1, self.label1, self._index_values(self.data1))
        data2_ , label2_ = self._pop_list(self.data2, self.label2, self._index_values(self.data2))

        for i in range(len(data1_)):
            concat_data.append(data1_[i])
            concat_label.append(label1_[i])

        for j in range(len(data2_)):
            concat_data.append(data2_[j])
            concat_label.append(label2_[j])

        return shuffle(concat_data, concat_label)




data_clean = data_cleaning_concat(sliced_all_files_ad, sliced_all_files_cn,
                                                list(label_ad) , list(label_cn))
total_data , total_label = data_clean.files_concat()

def cross_valid_splits(data, label, fold):
    fold_train = []
    fold_test = []
    fold_ytrain = []
    fold_ytest = []

    indxs1 = int(1*len(data)/5)
    indxs2 = int(2*len(data)/5)
    indxs3 = int(3*len(data)/5)
    indxs4 = int(4*len(data)/5)
    indxs5 = int(5*len(data)/5)

    rn_trn = []
    rn_tst = []
    if fold == 1:
        for i in range(0, indxs4):
            fold_train.append(data[i])
            fold_ytrain.append(label[i])

        for j in range(indxs4, indxs5):
            fold_test.append(data[j])
            fold_ytest.append(label[j])

    elif fold == 2:
        for i in range(0, indxs3):
            fold_train.append(data[i])
            fold_ytrain.append(label[i])
        for i in range(indxs4,indxs5):
            fold_train.append(data[i])
            fold_ytrain.append(label[i])


        for j in range(indxs3, indxs4):
            fold_test.append(data[j])
            fold_ytest.append(label[j])

    elif fold == 3:
        for i in range(0, indxs2):
            fold_train.append(data[i])
            fold_ytrain.append(label[i])
        for i in range(indxs3,indxs5):
            fold_train.append(data[i])
            fold_ytrain.append(label[i])


        for j in range(indxs2, indxs3):
            fold_test.append(data[j])
            fold_ytest.append(label[j])


    elif fold == 4:
        for i in range(0, indxs1):
            fold_train.append(data[i])
            fold_ytrain.append(label[i])
        for i in range(indxs2,indxs5):
            fold_train.append(data[i])
            fold_ytrain.append(label[i])


        for j in range(indxs1, indxs2):
            fold_test.append(data[j])
            fold_ytest.append(label[j])


    elif fold == 5:
        for i in range(indxs1, indxs5):
            fold_train.append(data[i])
            fold_ytrain.append(label[i])

        for j in range(0, indxs1):
            fold_test.append(data[j])
            fold_ytest.append(label[j])


    return fold_train, fold_test, np.array(fold_ytrain), np.array(fold_ytest)


f_train1, f_test1, f_ytrain1, f_ytest1 = cross_valid_splits(total_data, total_label,1)

f_train2, f_test2, f_ytrain2, f_ytest2 = cross_valid_splits(total_data, total_label,2)

f_train3, f_test3, f_ytrain3, f_ytest3 = cross_valid_splits(total_data, total_label,3)

f_train4, f_test4, f_ytrain4, f_ytest4 = cross_valid_splits(total_data, total_label,4)

f_train5, f_test5, f_ytrain5, f_ytest5 = cross_valid_splits(total_data, total_label,5)