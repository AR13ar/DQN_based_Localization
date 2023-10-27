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
import torch

f_train1 = "path to Training samples"
f_ytrain1 = "path to Training labels"
f_test1 = "path to Test samples"
f_ytest1 = "path to Test labels"


class data_set(Dataset):
  def __init__(self, file_path, label, transform_train):
    ### Variable definition to be used in Data loader
    self.file_path = file_path
    self.label = label
    self.transform_train = transform_train

  def __len__(self):
    ### size of file path
    return len(self.file_path)

  def __getitem__(self, indx):
    '''
      Read the image given indx from len function
    '''
    image_input = np.ones((145,256,256))
    #for j in range(indx):
    for i in range(145):
        try:
            img, _ = load(self.file_path[indx][i])
            img = cv2.resize(img.squeeze(), (256, 256))
            img = self.transform_train(img.squeeze().astype('float'))
            image_input[i,:,:] = img
        except:
            print("Check Index: ", indx)
    box_gd = np.array([np.array(self.label)[indx][0], np.array(self.label)[indx][1], np.array(self.label)[indx][2],
              np.array(self.label)[indx][3], np.array(self.label)[indx][4], np.array(self.label)[indx][5]])

    label_ = np.array(self.label)[indx][6]
    return [image_input, box_gd, label_]



train_data = data_set(file_path = f_train1, label = f_ytrain1, transform_train = transforms.Compose([
                                                                                   transforms.ToTensor(),
                                                                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                                                   ]))

unseen_data = data_set(file_path = f_test1, label = f_ytest1, transform_train = transforms.Compose([
                                                                                   transforms.ToTensor(),
                                                                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                                                   ]))

dataloader_train = torch.utils.data.DataLoader(train_data, batch_size = 64,
                        shuffle= True, num_workers= 0)
dataloader_unseen = torch.utils.data.DataLoader(unseen_data, batch_size = 1,
                        shuffle= True, num_workers= 0)