import numpy as np
import os, sys

state = [np.load("./input_data/state.npy")]
action = [np.load("./input_data/action.npy")]
reward = [ np.load("./input_data/reward.npy")]
for i in range(4):
    state.append(np.load("state_nonzero"+str(i+1)+".npy"))
    action.append(np.load("action_nonzero"+str(i+1)+".npy"))
    reward.append(np.load("reward_nonzero"+str(i+1)+".npy"))
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
print("bundling as final data")
state_f = np.concatenate(state_shuffle, axis = 0)
action_f = np.concatenate(action_shuffle, axis = 0)
reward_f = np.concatenate(reward_shuffle, axis = 0)
# delete big data
print("Delete Big data")
dirname = "./input_data/"
filename = "state.npy"
pathname = os.path.abspath(os.path.join(dirname, filename))
if pathname.startswith(dirname):
   os.remove(pathname)


print("Started to Save Data")
np.save("state.npy", state_f)
np.save("action.npy", action_f)
np.save("reward.npy", reward_f)
