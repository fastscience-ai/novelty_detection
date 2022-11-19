from obs_to_imgs import *
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import shutil
from ab_dataset_tensor import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from ab_fcnn import *
import pickle
import agent

def convert_pickle_to_image(path_to_pickle):
    observation = pickle.load(open(path_to_pickle, "rb"))

    obsimg = SBObs_to_Imgs()
    state, action, inter_states = obsimg.Obs_to_StateActionNextState(observation)
    reward =  observation.reward
    state_img, save = obsimg.state_to_nD_img(state)
    state_img =  np.reshape(state_img, [1, 480, 840, 13])
    state_img = state_img[:,100:400,0:600,:] #####################################################
    action = np.reshape(action, [1, 5])
    reward = np.reshape(reward, [1, 1])
    # input size for torch.nn.Conv2d : (N, C, W, H)
    state_img = np.transpose(state_img, (0,3,1,2))
    #print(np.shape(state_img), np.shape(action), np.shape(reward)) #(1, 13, 480, 840) (1, 5) (1, 1)
    return state_img, action, reward

def collect_data(path):
    # path: ./data/200_level_1_type_9_novelties/
    #       ./data/200_level_1_type_10_novelties/
    # input size for torch.nn.Conv2d : (N, C, W, H)
    state_input = []
    action_input = []
    reward_output = []
    for subdir1 in os.listdir(path):
        for subdir2 in os.listdir(path+"/"+subdir1):
            print("collect data from   : "+subdir1+"/"+subdir2)
            for idx, file in enumerate(glob.glob1(os.path.join(os.path.join(path, subdir1), subdir2), '*.p')):
                print("Collecting file at "+path+"/"+subdir1+"/"+subdir2+"/"+file)
                state_img, action, reward = convert_pickle_to_image(path+"/"+subdir1+"/"+subdir2+"/"+file)
                state_input.append(state_img)
                action_input.append(action)
                reward_output.append(reward)
    state_input = np.concatenate(state_input, axis = 0)
    action_input = np.concatenate(action_input, axis = 0)
    reward_output = np.concatenate(reward_output, axis = 0)
    print(np.shape(state_input), np.shape(action_input), np.shape(reward_output))

    #Shuffle
    assert(len(state_input)  == len(action_input) == len(reward_output))
    idx = [ i for i in range(len(state_input))]
    state_input_s = []
    action_input_s = []
    reward_output_s = []
    for i in range(len(state_input)):
        state_input_s.append(state_input[i:i+1])
        action_input_s.append(action_input[i:i+1])
        reward_output_s.append(reward_output[i:i+1])

    state_input_s = np.concatenate(state_input_s, axis = 0)
    action_input_s = np.concatenate(action_input_s, axis = 0)
    reward_output_s = np.concatenate(reward_output_s, axis = 0)   
    path_in = "./soo_novelty_detection/input_data/"
    np.save(path_in+"state.npy", state_input_s)
    np.save(path_in+"action.npy", action_input_s)
    np.save(path_in+"reward.npy", reward_output_s)
        
    print(np.shape(state_input_s), np.shape(action_input_s), np.shape(reward_output_s))
    return state_input_s, action_input_s, reward_output_s

   
    
def collect_data_from_folder(path):
    # input size for torch.nn.Conv2d : (N, C, W, H)
    state_input = []
    action_input = []
    reward_output = []
    print("Collecting file at "+path)
    for subdir in os.listdir(path):
        for idx, file in enumerate(glob.glob1(os.path.join(path, subdir), "*.p")):
            print(path+"/"+subdir+"/"+file)
            state_img, action, reward = convert_pickle_to_image(path+"/"+subdir+"/"+file)
            state_input.append(state_img)
            action_input.append(action)
            reward_output.append(reward)      
    state_input = np.concatenate(state_input, axis = 0)
    action_input = np.concatenate(action_input, axis = 0)
    reward_output = np.concatenate(reward_output, axis = 0)
    return state_input, action_input, reward_output    
    
def collect_reward(path):
    # path: ./data/200_level_1_type_9_novelties/
    #       ./data/200_level_1_type_10_novelties/
    # input size for torch.nn.Conv2d : (N, C, W, H)
    reward_output = []
    print("Collecting file at "+path)
    for idx, file in enumerate(glob.glob1(path, "*.p")):
        #print("Collecting file at "+path+"/"+file)
        _, _, reward = convert_pickle_to_image(path+"/"+file)
        reward_output.append(reward)

    reward_output = np.concatenate(reward_output, axis = 0)
    return  reward_output    

def normalization_reward(reward_output):
    reward_max = [4216.0] #[100000.0]
    reward_min = [0.0]

    np.save("reward_max.npy", np.asarray(reward_max))
    np.save("reward_min.npy", np.asarray(reward_min))
    print("Normalization start")

    reward_output = (reward_output[:, 0] - reward_min[0])/(reward_max[0] - reward_min[0])
    #with open("state_input_1_1.pkl",'wb') as f:
    #    pickle.dump( state_input, f)
    #with open("action_input_1_1.pkl",'wb') as f:
    #    pickle.dump( action_input, f)
    #with open("reward_output_1_1.pkl",'wb') as f:
    #    pickle.dump( reward_output, f)   

    #np.savez("nomarlized_level0_2.npz", state = state_input, action = action_input, reward = reward_output)
    return reward_output


def normalization(state_input, action_input, reward_output):
    state_max = [np.amax(state_input[:,i,:,:]) for i in range(13)]
    state_min = [np.amin(state_input[:,i,:,:]) for i in range(13)]
    #action_max = [-8, 225, 3000, 160, 326] 
    action_max = [np.amax(action_input[:,i]) for i in range(5)]
    #action_min = [-226, 0, 3000, 142, 318]
    action_min = [np.amin(action_input[:,i]) for i in range(5)]
    #if np.amax(reward_output) != np.amin(reward_output):
    #    reward_max = [np.amax(reward_output)]
    #    reward_min = [np.amin(reward_output)]
    #else:
    #    reward_max = [42160.0]
    #    reward_min = [0.0]
    print(np.amax(reward_output), np.amin(reward_output))
    #if np.amax(reward_output) > 42160.0:
    reward_max = [np.amax(reward_output)]
    reward_min = [0.0]
    #else:
    #    reward_max = [42160.0]
    #    reward_min = [0.0]
    #print("reward_max !!!!!!!!!!!!!!!!!!")
    #print(reward_max)

    
    np.save("state_max.npy", np.asarray(state_max))
    np.save("state_min.npy", np.asarray(state_min))
    np.save("action_max.npy", np.asarray(action_max))
    np.save("action_min.npy", np.asarray(action_min))
    np.save("reward_max.npy", np.asarray(reward_max))
    np.save("reward_min.npy", np.asarray(reward_min))
    print("Normalization start")
    for i in range(13):
        if state_max[i] != state_min[i]:
            state_input[:,i,:,:] = (state_input[:,i,:,:] - state_min[i]) / (state_max[i] - state_min[i])
        else:
            state_input[:, i, :, :] = 0.0


    for j in range(5):
        if action_max[j] != action_min[j]:
             action_input[:,j] = (action_input[:,j] - action_min[j])/ (action_max[j]-action_min[j])
        else:
            action_input[:,j] = 0.0
    reward_output = (reward_output[:, 0] - reward_min[0])/(reward_max[0] - reward_min[0])
    reward_output = np.reshape(reward_output, [-1,1])
 
    
    #with open("state_input_nonzero.pkl",'wb') as f:
    #    pickle.dump( state_input, f)
    #with open("action_input_nonzero.pkl",'wb') as f:
    #    pickle.dump( action_input, f)
    #with open("reward_output_nonzero.pkl",'wb') as f:
    #    pickle.dump( reward_output, f)   

    np.save("./soo_novelty_detection/input_data/state.npy", np.asarray(state_input))
    np.save("./soo_novelty_detection/input_data/action.npy", np.asarray(action_input))
    np.save("./soo_novelty_detection/input_data/reward.npy", np.asarray(reward_output))
    print(np.shape(state_input), np.shape(action_input), np.shape(reward_output))
    #np.savez("nomarlized_level0.npz", state = state_input, action = action_input, reward = reward_output)
    return state_input, action_input, reward_output


def normalization_L0(state_input, action_input, reward_output):
    path="./soo_novelty_detection/input_data/" # Path of saving data files of L0
    state_max = np.load(path+"state_max.npy")
    state_min = np.load(path+"state_min.npy")
    action_max = np.load(path+"action_max.npy")
    action_min = np.load(path+"action_min.npy")
    reward_max = np.load(path+"reward_max.npy")
    reward_min = np.load(path+"reward_min.npy")
    print(np.amax(reward_output), np.amin(reward_output))
    print("Normalization start")
    for i in range(13):
        if state_max[i] != state_min[i]:
            state_input[:,i,:,:] = (state_input[:,i,:,:] - state_min[i]) / (state_max[i] - state_min[i])
        else:
            state_input[:, i, :, :] = 0.0

    for j in range(5):
        if action_max[j] != action_min[j]:
             action_input[:,j] = (action_input[:,j] - action_min[j])/ (action_max[j]-action_min[j])
        else:
            action_input[:,j] = 0.0
    reward_output = (reward_output[:, 0] - reward_min[0])/(reward_max[0] - reward_min[0])
    reward_output = np.reshape(reward_output, [-1,1])
    print(np.shape(state_input), np.shape(action_input), np.shape(reward_output))
    return state_input, action_input, reward_output

