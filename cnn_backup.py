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
            #Conv2d(13, 26, kernel_size=3, stride=2, padding=1),
            #BatchNorm2d(26),
            #ReLU(inplace=True),
            #MaxPool2d(kernel_size=2, stride=1),
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
            Linear(  3072, 1024), #757248 # 358400]
        )

        self.u_enc = Linear(5, 1024, bias=False)

        self.fc = Sequential(
            #Linear(1024, 2048), ===============================>correct
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
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.linear_layers(x)
      #  print(x.shape, u.shape) #torch.Size([6, 1024]) torch.Size([6, 5])
        x = torch.cat((x,u),1)#============================================================>correct
        #print(x.shape)
        #x = self.fc(x*self.u_enc(u))
        #print(x.shape)
        x = self.fc(x)
        #x = self.fc(x*self.u_enc(u))=================================================>correct
       # print(x.shape)
        return x

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
    model = CNN()
    optimizer = Adam(model.parameters(), lr=0.0000005)
    # defining the loss function
    criterion = MSELoss()
    # checking if GPU is available
    print(torch.cuda.is_available())
    #if torch.cuda.is_available():
    #    free_gpu_cache()
    #   free_gpu_cache()
    #    model = model.cuda()
    #    criterion = criterion.cuda()
    #    print("GPU usage after putting our model: ")
    #    gpu_usage()

    print(model)
    #trouble shooting
    #model(torch.randn(6, 13, 300, 600).cuda(), torch.randn(6,5).cuda()).float().cuda()

    #load data
    state_input, action_input, reward_output = collect_data("./data/211007_100_level_0_type_2_novelties/100_level_0_type_2_novelties/")
    state_input, action_input, reward_output = normalization(state_input, action_input, reward_output)
    data = np.load("nomarlized_obs_200_level_0_type_2_novelties.npz")
    state_input = data['state'][:,:,100:400,0:600]
    action_input = data['action']
    reward_output = np.reshape(data['reward'], [-1,1])
    print(np.shape(state_input), np.shape(action_input), np.shape(reward_output))
    print(np.amax(state_input), np.amin(state_input), np.amax(action_input), np.amin(action_input), np.amax(reward_output), np.amin(reward_output))

    n = len(state_input)
    x_train_1, x_test_1 = state_input[:int(n*0.98)], state_input[int(n*0.98):]
    x_train_2, x_test_2 = action_input[:int(n * 0.98)], action_input[int(n * 0.98):]
    y_train, y_test = reward_output[:int(n * 0.98)] , reward_output[int(n * 0.98):]
    print(np.shape(x_train_1), np.shape(x_train_2), np.shape(x_test_1), np.shape(x_test_2), np.shape(y_train), np.shape(y_test))
    N, C, H, W = np.shape(x_train_1)
    f_log =open("log_concat.txt", "w")
    batch_size = 20
    if state_file is None: #train
        #train
        epoch_num =500
        model.train()
        os.mkdir("./saved_model_reward_prediction")
        for epoch in range(epoch_num+1):
            running_loss = 0.0
            for i in range(int(N/batch_size)):
                optimizer.zero_grad()
                output = model(torch.tensor(x_train_1[i*batch_size: (i+1)*batch_size,:,:,:]).float().cuda(), torch.tensor(x_train_2[i*batch_size: (i+1)*batch_size,:]).float().cuda())
                loss = criterion(output, torch.tensor(y_train[i*batch_size: (i+1)*batch_size,:]).float().cuda())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i%10 == 0:
                        print("loss of {} epoch  , {} index out of {} : {}".format(epoch, i, N/batch_size, running_loss/float(i+1)))
                        f_log.write("loss of {} epoch  , {} index out of {} : {}".format(epoch, i, N/batch_size, running_loss/float(i+1)))
                        f_log.write("\n")
                if epoch % 100 ==0 and epoch > 0 :
                        for j in range(int(N*0.02/batch_size)):
                            y_hat = model(torch.tensor(x_test_1[j*batch_size: (j+1)*batch_size, :,:,:]).float().cuda(),torch.tensor(x_test_2[j*batch_size: (j+1)*batch_size, :]).float().cuda())
                            val_loss = criterion(y_hat, torch.tensor(y_test[j*batch_size: (j+1)*batch_size, :]).float().cuda()).item()
                            print("val Loss (total epoch"+str(int(epoch))+"): "+str(val_loss))
                            f_log.write("\n")
                            f_log.write("val Loss (total epoch"+str(int(epoch))+"): "+str(val_loss))
                            f_log.write("\n")
                            y_hat  =  y_hat.cpu().detach().numpy()
                        PATH = './saved_model_reward_prediction/'+str(epoch)+'.pth'
                        torch.save(model.state_dict(), PATH)
    else:
         model.load_state_dict(torch.load(state_file))
         # evaluate accuracy on the training data
         #Y_pred = []
         #for j in range(len(x_test_2)-1):
         #   y_hat = model(torch.tensor(x_test_1[j: (j+1), :,:,:]).float().cuda(),torch.tensor(x_test_2[j: (j+1), :]).float().cuda())
         #   print(np.shape(y_hat), str(j+1)+"-th test sample")
         #   Y_pred.append(y_hat)
         #Y_pred = np.concatenate(Y_pred, axis = 0)
         Y_pred = model(torch.tensor(x_test_1).float(),torch.tensor(x_test_2).float())
         Y_pred = Y_pred.detach().numpy()
         print(np.shape(y_test),np.shape(Y_pred))
         mse = np.mean( (y_test - Y_pred)**2)
         print("Total MSE: "+str(mse))
         y_max = np.load("reward_max.npy")
         y_min = np.load("reward_min.npy")
         os.mkdir("reward_prediction_mse_"+str(mse*(y_max[0]-y_min[0])+y_min[0]))
         
         print("Total MSE after denormalization: "+str(mse*(y_max[0]-y_min[0])+y_min[0]))
         print(y_test.shape)
         fout = open("reward_prediction_mse_" + str(mse*(y_max[0]-y_min[0])+y_min[0]) + "/output.csv", "w")
         fout.write("ground truth, prediction" + "\n")
         gt =[]
         pr = []
         for j in range(y_test.shape[0]):
             fout.write(str(y_test[j,0]*(y_max[0]-y_min[0])+y_min[0])+ ", "+str(Y_pred[j,0]*(y_max[0]-y_min[0])+y_min[0])+"\n")
             gt.append(float(y_test[j,0]*(y_max[0]-y_min[0])+y_min[0]))
             pr.append(float(Y_pred[j,0]*(y_max[0]-y_min[0])+y_min[0]))
         plt.scatter(gt, pr, marker='.')
         plt.xlabel("Ground Truth")
         plt.ylabel("Prediction")
         plt.title("train with L0 novelty, test with L0 "+", MSE :"+str(mse*(y_max[0]-y_min[0])+y_min[0]))
         plt.savefig("reward_prediction_mse_" + str(mse*(y_max[0]-y_min[0])+y_min[0])+"/"+"fig.png")
         plt.close()
         fout.close()
    f_log.close()

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(10)
    epoch = 100
    state_file ='./saved_model_reward_prediction/'+str(epoch)+'.pth'
    #state_file = None
    main(state_file) # if state_file is None: training, otherwise, eval of pretraine-model
