import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d,MaxUnpool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from skimage import util
from torchvision import transforms, models
from torch import optim

import numpy as np
import random
import torchvision
from collections import namedtuple, deque
import math

class DQNSolver(nn.Module):
    """
    Convolutional Neural Net with 3 conv3D layers and two linear layers
    """
    def __init__(self, input_shape, n_actions):
        super(DQNSolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(input_shape[0], 32, kernel_size=(5,5,5), stride=2),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(5,5,5), stride=1),
            nn.LeakyReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=2),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

policy_model = DQNSolver((1,50,256,256), 7)
target_model = DQNSolver((1,50,256,256), 7)


class Classifier(nn.Module):
    """
    Convolutional Neural Net with 3 conv3D layers and two linear layers
    """
    def __init__(self, input_shape, num_features):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(input_shape[0], 32, kernel_size=(5,5,5), stride=2),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(5,5,5), stride=1),
            nn.LeakyReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=2),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_features),
            nn.ReLU()
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

classifier_hippocampus = Classifier((1,50,256,256), 512)
classifier_global = Classifier((1,145,256,256), 1024)

class concat_classifier(nn.Module):
    def __init__(self, input_shape1,input_shape2, classes):
        super(concat_classifier, self).__init__()

        input_shape = input_shape1[1] + input_shape2[1]
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, classes),
            nn.Sigmoid()
            )

    def forward(self,x1, x2):
        cat_out = torch.cat((x1,x2), dim = 1)
        output = self.fc(cat_out)
        return output

final_model = concat_classifier((1,512), (1, 1024), 2)

class combine_model(nn.Module):
    def __init__(self, model1, model2, model3):
        super(combine_model, self).__init__()

        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def forward(self, x1, x2):
        input1 = x1
        input2 = x2

        out1 = self.model1(input1)
        out2 = self.model2(input2)
        pred = self.model3(out1, out2)

        return pred

class_model = combine_model(classifier_hippocampus, classifier_global, final_model)