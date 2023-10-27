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
from sklearn.metrics import confusion_matrix
import cv2
import torchvision
import math
from sklearn.utils import shuffle
from tqdm import tqdm

def testing_phase(dqn_model,classifier_model, dataloader, criterion):
    testing_acc = []
    testing_loss = []
    pred_test = []
    pred_true = []
    final_box = []
    test_loss = 0.0
    test_acc = 0.0
    total = 0.0
    for data in dataloader:
        images, box_gds, labels = data
        batches, depth, height, width = images.shape
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
            for episodes in range(500):
                eps = max(0,(start_eps/int(1 + episodes)   - end_eps))
                step = max(end_step, start_step/int(1 + episodes) )

                prev_patch = crop_reshape(env, prev_state[0],prev_state[1],prev_state[2],prev_state[3],prev_state[4],prev_state[5])
                action = dqn_model.select_action_test(prev_patch)
                if action != 6:
                    nxt_state = dqn_model.next_state(prev_state, action, step)
                    # print(nxt_state)
                    nxt_patch = crop_reshape(env, nxt_state[0],nxt_state[1],nxt_state[2],nxt_state[3],nxt_state[4],nxt_state[5])
                    game, reward = dqn_model.compute_reward(nxt_state, prev_state, state_gd, 10)
                else:
                    game = "END"
                    nxt_state = prev_state.copy()
                    nxt_patch = crop_reshape(env, nxt_state[0],nxt_state[1],nxt_state[2],nxt_state[3],nxt_state[4],nxt_state[5])
                    _, reward = dqn_model.compute_reward(nxt_state,prev_state, state_gd, 10)
                    #print(np.asarray(action))
                prev_state = nxt_state.copy()
                prev_patch = crop_reshape(env, nxt_state[0],nxt_state[1],nxt_state[2],nxt_state[3],nxt_state[4],nxt_state[5])
                rewards += reward
                #print("Episodes : ", episodes)
                if game == "END" or episodes == 500:
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
        out_pred = classifier_model(dqn_mask.float(), images.view(batches, 1, depth, height, width).float())
        loss = criterion(out_pred, labels.type(torch.LongTensor))#.view(-1,1) float()
        test_loss += loss.item()
        total += labels.size(0)
        test_acc +=  binary_acc(out_pred, labels)
        pred_test.append(int(out_pred.detach()))
        pred_true.append(int(labels))
        final_box.append([nxt_state, np.array(state_gd)])
    test_loss = test_loss/len(dataloader)
    testing_loss.append(test_loss)
    test_acc = test_acc/total #100 *
    testing_acc.append(test_acc)
    print('Test Loss: {:.6f} \tTest Accuracy: {:.6f}'.format( test_loss, test_acc))
    return pred_test, final_box, test_loss, test_acc, pred_true


a,b,c,d,e = testing_phase(my_dqn, class_model,  dataloader_unseen, criterion )


def bar_plot(output_pred, output_true, unseen_loss, unseen_acc):
  plt.figure(2)
  cm = confusion_matrix(np.array(output_pred).squeeze(), np.array(output_true).squeeze())
  recall = 100*cm[0,0]/(cm[0,0] + cm[1,0])
  precision = 100*cm[0,0]/(cm[0,0] + cm[0,1])
  params = ['Loss', 'Accuracy', 'TN', 'FP', 'FN', 'TP', 'Precision', 'Recall']
  results = [unseen_loss, unseen_acc,cm[1,1],cm[0,1],cm[1,0],cm[0,0], precision, recall]
  plt.bar(params, results)
  for index,data in enumerate(results):
    plt.text(x = index -0.2 , y = data  , s = f"{int(data)}" , fontdict=dict(fontsize=10))
  plt.tight_layout()
  plt.show()

bar_plot(a,e,c,100*d)