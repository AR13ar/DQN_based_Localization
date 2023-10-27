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

from medpy.io import load
import cv2

''' DQN Agent
'''
class DQNAgent(object):
    def __init__(self, policy_model, target_model, replay_memory):
            self.batch_size = 100
            self.gamma = 0.99
            self.eps = 1
            self.target_update = 0
            screen_height, screen_width, screen_depth = 256, 256, 145
            self.n_actions = 7
            self.policy_model = policy_model
            self.target_model = target_model
            self.target_model.eval()
            self.optimizer = optim.Adam(self.policy_model.parameters(), lr = 0.00009)
            self.loss_func = torch.nn.SmoothL1Loss()#nn.MSELoss()
            self.memory = replay_memory

    ''' Compute reward with L2 distance between centers of state and gd
    '''
    def compute_reward(self, actual_state, prev_state, ground_truth, threshold):
            x,y,z,w,h,d = actual_state
            x_p,y_p,z_p,w_p,h_p,d_p = prev_state
            x_gd, y_gd, z_gd, w_gd, h_gd, d_gd = ground_truth

            center = np.array([(x+w)/2,(y+h)/2, (z+d)/2])
            center_gd = np.array([(x_gd + w_gd)/2,(y_gd + h_gd)/2, (z_gd + d_gd)/2])
            center_p = np.array([(x_p + w_p)/2,(y_p + h_p )/2, (z_p + d_p)/2])
            dist_l2 = np.linalg.norm(center - center_gd)
            dist_l2_p = np.linalg.norm(center_p - center_gd)

            if dist_l2 < threshold:
                game = "END"
                reward = 100
                return game, reward
            else:
                game = "continue"
                reward = min(1, -dist_l2) #dist_l2 - dist_l2_p #
                return game, reward

    ''' 0 - up, 1- down, 2 - right, 3- left, 4- top, 5 - bottom, 6 - terminate
    '''
    def next_state(self, prev_state, actn, step):
            x,y,z,w,h,d = prev_state
            max_x = 255
            max_y = 255
            max_z = 144
            min_x = 0
            min_y = 0
            min_z = 0

            if actn == 0:
                if x + w + step >= 255:
                    x = 100
                else:
                    x = x + step
            elif actn == 1:
                if x - step <= 0:
                    x = 100
                else:
                    x = x - step
            elif actn == 2:
                if y + h + step >= 255:
                    y = 100
                else:
                    y = y + step
            elif actn == 3:
                if y - step <= 0:
                    y = 100
                else:
                    y = y - step
            elif actn == 4:
                if z + d + step >= 144 :
                    z = 50
                else:
                    z = z + step
            elif actn == 5:
                if z - step <= 0:
                    z = 50
                else:
                    z = z - step

            return [int(x),int(y),int(z),int(w),int(h),int(d)]

    ''' Start with high eps then reduce eps, if game = END then action should be 6
    '''
    def select_action(self, state, game, eps ):
            actn = 6
            if game == "continue":
                sample = random.random()
                if sample < eps:
                    actn = np.asarray(random.randrange(6))

                else:
                    out = self.policy_model(state)
                    _, actn = torch.max(out.data, 1)

                actn = np.array(actn)
                return actn
            else:
                actn = np.array(actn)
                return actn

    def select_action_test(self, state):
        out = self.target_model(state)
        _, actn = torch.max(out.data,1)
        return actn

    def store_transition(self, s, a, r, s_):
            self.memory.store(s, a, r, s_)

    ''' Optimize model with replay memory
    '''

    def optimize(self, counter):

            if self.target_update > counter:
                self.target_model.load_state_dict(self.policy_model.state_dict())
                self.target_update = 0
                #print("Updating Target Net")

            self.target_update += 1


            s_s, a_s, r_s, s__s = self.memory.sample(self.batch_size)

            q_eval = self.policy_model(s_s).gather(1, a_s)
            q_next = self.target_model(s__s).detach()
            q_target = r_s + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
            loss = self.loss_func(q_eval, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss

my_dqn = DQNAgent(policy_model, target_model, ReplayMemory(1000))