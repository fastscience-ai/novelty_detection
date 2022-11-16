# importing the libraries
import pandas as pd
import numpy as np
# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt
# for creating validation set
from sklearn.model_selection import train_test_split
# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm
# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, MSELoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from data import *
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import sys, os
#Free GPU memory
import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda


class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(13, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=4, stride=2),
            # Defining another 2D convolution layer
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=4, stride=2),
            # Defining another 2D convolution layer
            Conv2d(128, 256, kernel_size=5, stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=4, stride=2),
            Conv2d(256, 512, kernel_size=5, stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=4, stride=2),
            Conv2d(512, 1024, kernel_size=5, stride=2, padding=1),
            BatchNorm2d(1024),
            ReLU(inplace=True),
        )
        self.linear_layers = Sequential(
            Linear(  3072, 1024), 
        )
        self.u_enc = Linear(5, 1024, bias=False)
        self.fc = Sequential(
            Linear(1029, 2048),
            ReLU(),
            Linear(2048, 4096),
            ReLU(),
            Linear(4096, 8192),
            ReLU(),
            Linear(8192, 1),
            ReLU()
        )

    # Defining the forward pass
    def forward(self, x, u):
        x = self.cnn_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear_layers(x)
        x = torch.cat((x,u),1)
        x = self.fc(x)
        return x


def test_one_pickle_file(model, state_file, path_to_pickle):
    # load data
    state_input, action_input, reward_output = convert_pickle_to_image(path_to_pickle)
    state_input, action_input, reward_output = normalization(state_input, action_input, reward_output)
    reward_output = np.reshape(reward_output, [-1,1])    
    # predict reward
    y_hat = model(torch.tensor(state_input).float(),torch.tensor(action_input).float())
    print("(1) action :" + str(action_input[0,:]))
    print("(2) ground truth reward:" + str(reward_output[0,:]))
    print("(3) predicted reward: "+str(y_hat[0,:].detach().numpy()))
    return state_input, action_input, reward_output, y_hat   
    
def test_one_pickle_file_with_customized_action(model, state_file, path_to_pickle, action):
    # load data
    state_input, _, reward_output = convert_pickle_to_image(path_to_pickle)
    action_input = np.asarray([action])
    state_input, action_input, reward_output = normalization(state_input, action_input, reward_output)
    reward_output = np.reshape(reward_output, [-1,1])    
    # predict reward
    y_hat = model(torch.tensor(state_input).float(),torch.tensor(action_input).float())
    print("(1) customized action (Taken as the input and normalized btw [0,1]):" + str(action_input[0,:]))
    print("(2) ground truth reward:" + str(reward_output[0,:]))
    print("(3) predicted reward: "+str(y_hat[0,:].detach().numpy()))
    return state_input, action_input, reward_output, y_hat
    
def main(state_file, path_to_pickle):
    #1. Loading pretrained model
    model = CNN()
    model.load_state_dict(torch.load(state_file)) 
    #2. Take input (state, action) from pickle file and predict reward using pre-trained model.
    print("Example test 1. Take input (state, action) from pickle file and predict reward using pre-trained model.")
    test_one_pickle_file(model, state_file, path_to_pickle)
    action = [0,100,3000,155,320]
    #3. Take input (state) from pickle file, take customized action, and predict reward using pre-trained model.
    print("Example test 2. Take input (state) from pickle file, take customized action, and predict reward using pre-trained model.")
    test_one_pickle_file_with_customized_action(model, state_file, path_to_pickle, action)

if __name__ == '__main__':
    epoch = 100
    state_file ='./saved_model_reward_prediction/'+str(epoch)+'.pth'
    path_to_pickle = './9_3_211011175902_211011_181531_observation.p'
    main(state_file, path_to_pickle) 
