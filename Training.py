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
import torchvision
import math
from tqdm import tqdm
import numpy as np

def binary_acc(y_pred, y_test):
    #y_pred[y_pred >= 0.5] = 1 
    #y_pred[y_pred < 0.5] = 0
    _, y_pred = y_pred.max(1)
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum#/y_test.shape[0]
    acc = torch.round(acc)

    return acc

def training_phase(dqn_model, classifier_model, dataloader, criterion, optimizer):
    training_acc = []
    training_loss = []
    for epoch in range(100):
        train_loss = 0.0
        train_acc = 0.0
        total = 0.0
        for data in dataloader:
            images, box_gds, labels = data
            batches, depth, height, width = images.shape
            dqn_mask = []
            loss_batch = torch.tensor(np.ones(batches))
            reward_batch = torch.tensor(np.ones(batches))
            dqn_mask = torch.tensor(np.ones((batches,1,50,height,width)))
            for batch in range(batches):
                x_start = np.random.randint(80,120)
                y_start = np.random.randint(60,200-50)
                z_start = np.random.randint(50,145-50)
                w, h, d = 50, 60, 50
                start_state = [x_start, y_start, z_start, w, h, d]
                game = "continue"
                env = images[batch]
                x_gd, y_gd, z_gd = box_gds[batch][0], box_gds[batch][1], box_gds[batch][2]
                w_gd, h_gd, d_gd = box_gds[batch][3], box_gds[batch][4], box_gds[batch][5]
                state_gd = [x_gd, y_gd, z_gd, w_gd, h_gd, d_gd]
                start_eps = 1
                end_eps = 0.004
                start_step = 20
                end_step = 5
                prev_state = start_state.copy()
                loss = 0
                rewards = 0
                for episodes in range(100):
                    eps = max(0,(start_eps/int(1 + episodes)   - end_eps))
                    step = max(end_step, start_step/int(1 + episodes) )

                    prev_patch = crop_reshape(env, prev_state[0],prev_state[1],prev_state[2],prev_state[3],prev_state[4],prev_state[5])
                    action = dqn_model.select_action(prev_patch, game, eps)
                    if action != 6:
                        nxt_state = dqn_model.next_state(prev_state, action, step)
                        # print(nxt_state)
                        nxt_patch = crop_reshape(env, nxt_state[0],nxt_state[1],nxt_state[2],nxt_state[3],nxt_state[4],nxt_state[5])
                        game, reward = dqn_model.compute_reward(nxt_state, prev_state, state_gd, 10)
                    else:
                        game = "END"
                        nxt_state = prev_state.copy()
                        nxt_patch = crop_reshape(env, nxt_state[0],nxt_state[1],nxt_state[2],nxt_state[3],nxt_state[4],nxt_state[5])
                        _, reward = dqn_model.compute_reward(nxt_state, prev_state, state_gd, 10)
                    #print(np.asarray(action))
                    dqn_model.store_transition(prev_patch, np.asarray(int(action)), reward, nxt_patch)

                    if dqn_model.memory.memory_counter >= 1000:
                        loss = dqn_model.optimize(1500)


                    prev_state = nxt_state.copy()
                    prev_patch = crop_reshape(env, nxt_state[0],nxt_state[1],nxt_state[2],nxt_state[3],nxt_state[4],nxt_state[5])
                    rewards += reward
                    #print("Episodes : ", episodes)
                    if game == "END" or episodes == 2000:
                        dqn_mask[batch] = nxt_patch.view(1, nxt_patch.shape[-3], nxt_patch.shape[-2],nxt_patch.shape[-1])
                        loss_batch[batch] = (loss/int(1 + episodes))
                        reward_batch[batch] = (rewards/int(1 + episodes))
                        break
                print("State Pred {:} , ground truth {:}, episodes {:}, batch {:}".format(nxt_state , np.array(state_gd), episodes, batch))
            try:
                loss_per_batch = loss_batch.numpy().detach()
            except:
                loss_per_batch = loss_batch

            # print(loss_per_batch.shape)
            print("Loss_Rl {:.4f} \tReward_RL {:.4f}".format (loss_per_batch.mean(), reward_batch.mean()))

            optimizer.zero_grad()
            # print(dqn_mask.shape)
            out_pred = classifier_model(dqn_mask.float(), images.view(batches, 1, depth, height, width).float())
            # print(out_pred)
            # print(labels)
            loss = criterion(out_pred, labels.type(torch.LongTensor))#.view(-1,1), .float()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total += labels.size(0)
            train_acc +=  binary_acc(out_pred, labels)
            print(train_acc)
            print(total)
            # print(labels.size(0))
        train_loss = train_loss/len(dataloader)
        training_loss.append(train_loss)
        train_accuracy = train_acc/total #100 *
        training_acc.append(train_accuracy)
        print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.6f}'.format(epoch, train_loss, train_accuracy))
    return training_acc, training_loss

criterion = nn.CrossEntropyLoss()#nn.BCELoss()
optimizer = torch.optim.Adam(class_model.parameters(), lr= 0.00009,  weight_decay=1e-5)
t_phase1, l_phase1 = training_phase(my_dqn, class_model,  dataloader_train, criterion, optimizer )
