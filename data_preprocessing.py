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




#https://www.kaggle.com/getting-started/140636
def free_gpu_cache():
    print("Initial GPU Usage") 
    torch.cuda.empty_cache()
    gpu_usage()                             

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()
    
    
    
def main(state_file):


    if state_file is None: #train
        if True:
            path_to_data = "./data/120000pts/100_level_0_type_2_novelties/"
            state =[]
            action = []
            reward = []
            folder_name = ["r_3_1","r_2_2"] #"r_4_2",  
            for name in folder_name:
                state_input, action_input, reward_output = collect_data_from_folder(path_to_data+name+"/")
                reward_output = np.reshape(reward_output, [-1,1])
                state.append(state_input)
                action.append(action_input)
                reward.append(reward_output)
                print(np.shape(state_input), np.shape(action_input), np.shape(reward_output)) #(5242, 13, 300, 600) (5242, 5) (5242, 1)
                print(np.amax(state_input), np.amin(state_input), np.amax(action_input), np.amin(action_input), np.amax(reward_output), np.amin(reward_output))
            state = np.concatenate(state, axis = 0)
            action = np.concatenate(action, axis = 0)
            reward = np.concatenate(reward , axis = 0)
            print(np.shape(state), np.shape(action), np.shape(reward)) #(5242, 13, 300, 600) (5242, 5) (5242, 1)
            #Shuffle
            index = [i for i in range(len(reward))]
            import random
            random.shuffle(index)
            print(index)
            state_shuffle = []
            action_shuffle = []
            reward_shuffle = []
            for i in index:
                state_shuffle.append(state[i:i+1])
                action_shuffle.append(action[i:i+1])
                reward_shuffle.append(reward[i:i+1])
            state_f = np.concatenate(state_shuffle, axis = 0)
            action_f = np.concatenate(action_shuffle, axis = 0)
            reward_f = np.concatenate(reward_shuffle, axis = 0)
            np.save("state.npy", state_f)
            np.save("action.npy", action_f)
            np.save("reward.npy", reward_f)
       

   

            
            
            
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(10)

    state_file = None
    main(state_file) # if state_file is None: training, otherwise, eval of pretraine-model
