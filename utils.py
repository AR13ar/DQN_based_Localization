import numpy as np
import random
import cv2

from collections import namedtuple, deque
import math

''' Replay Buffer
'''
N_CHANNEL, N_HIGH, N_WEIGHT = 50, 256, 256
class ReplayMemory():
    def __init__(self, memory_size):
        self.memory_counter = 0
        self.memory_size = memory_size
        self.state_memory = torch.FloatTensor(self.memory_size, 1, N_CHANNEL, N_HIGH, N_WEIGHT)
        self.action_memory = torch.LongTensor(self.memory_size)
        self.reward_memory = torch.FloatTensor(self.memory_size)
        self.state__memory = torch.FloatTensor(self.memory_size, 1, N_CHANNEL, N_HIGH, N_WEIGHT)

    def store(self, s, a, r, s_):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = s
        self.action_memory[index] = torch.LongTensor([a.tolist()])
        self.reward_memory[index] = torch.FloatTensor([r])
        self.state__memory[index] = s_
        self.memory_counter += 1

    def sample(self, size):
        sample_index = np.random.choice(self.memory_size, size)
        state_sample = torch.FloatTensor(size,1,  N_CHANNEL, N_HIGH, N_WEIGHT)
        action_sample = torch.LongTensor(size, 1)
        reward_sample = torch.FloatTensor(size, 1)
        state__sample = torch.FloatTensor(size, 1, N_CHANNEL, N_HIGH, N_WEIGHT)
        for index in range(sample_index.size):
            state_sample[index] = self.state_memory[sample_index[index]]
            action_sample[index] = self.action_memory[sample_index[index]]
            reward_sample[index] = self.reward_memory[sample_index[index]]
            state__sample[index] = self.state__memory[sample_index[index]]
        return state_sample, action_sample, reward_sample, state__sample


''' Reshape state patches to full size input 50 x 256 x 256
'''

def crop_reshape(img, x,y,z,w,h,d):
    img = np.array(img)
    new_image = []
    image_patch = img[z:z+d, x:x+h, y:y+w]
    for i in range(image_patch.shape[0]):
        new_image.append(cv2.resize(image_patch[i], (256,256)))
    new_image = np.array(new_image)
    return torch.tensor(new_image.reshape(1,1,new_image.shape[0], new_image.shape[1], new_image.shape[2])).float()

'''Compute Accuracy for Training 
'''

def binary_acc(y_pred, y_test):
    #y_pred[y_pred >= 0.5] = 1 
    #y_pred[y_pred < 0.5] = 0
    _, y_pred = y_pred.max(1)
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum#/y_test.shape[0]
    acc = torch.round(acc)

    return acc
