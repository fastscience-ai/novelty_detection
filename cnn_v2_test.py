# importing the libraries
import pandas as pd
import numpy as np
import copy
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
            Linear( 3072, 1024), 
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
     #   print(x.shape)
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        
      #  print(x.shape)
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
    
    
    
def main(state_file, epoch):
    model = CNN()
    optimizer = Adam(model.parameters(), lr=0.0000001)
    # defining the loss function
    criterion = MSELoss()
    # checking if GPU is available
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        free_gpu_cache()
        free_gpu_cache()
        model = model.cuda()
        criterion = criterion.cuda()
        print("GPU usage after putting our model: ")
        gpu_usage()

    print(model)
    #trouble shooting
    model(torch.randn(6, 13, 300, 600).cuda(), torch.randn(6,5).cuda()).float().cuda()
    '''
    #load data
    state_input, action_input, reward_output = collect_data("./data/level0/")
    state_input, action_input, reward_output = normalization(state_input, action_input, reward_output)
    data = np.load("nomarlized_obs_200_level_0_type_2_novelties.npz")
    state_input = data['state'][:,:,100:400,0:600] # did alredy
    state_input = data['state']
    action_input = data['action']
    reward_output = np.reshape(data['reward'], [-1,1])
    print(np.shape(state_input), np.shape(action_input), np.shape(reward_output))
    print(np.amax(state_input), np.amin(state_input), np.amax(action_input), np.amin(action_input), np.amax(reward_output), np.amin(reward_output))
    '''
    path_in = "./input_data/"
    state_input, action_input, reward_output = np.load(path_in+"state.npy"), np.load(path_in+"action.npy"), np.load(path_in+"reward.npy")
    DATA = "l25t1" #"l1t7" "l2t9"
    #state_input_l1t7, action_input_l1t7, reward_output_l1t7 = np.load(path_in+"state_"+DATA+".npy"), np.load(path_in+"action_"+DATA+".npy"), np.load(path_in+"reward_"+DATA+".npy")
    if DATA == "l0t222to225": 
        state_input_l1t7, action_input_l1t7, reward_output_l1t7 = np.load(path_in+"state.npy"), np.load(path_in+"action.npy"), np.load(path_in+"reward.npy")
    else: 
        state_input_l1t7, action_input_l1t7, reward_output_l1t7 = np.load(path_in+"state_"+DATA+".npy"), np.load(path_in+"action_"+DATA+".npy"), np.load(path_in+"reward_"+DATA+".npy")

    n = len(state_input)
    x_train_1, x_test_1 = state_input[:int(n*0.9)], state_input[-100:]
    x_train_2, x_test_2 = action_input[:int(n*0.9)], action_input[-100:]
    y_train, y_test = reward_output[:int(n*0.9)] , reward_output[-100:]
    
    if DATA=="l2t9":
        x_test_1, x_test_2, y_test = x_test_1[:40] , x_test_2[:40], y_test[:40] #l2t9
    
    if DATA == "l0t222to225":
        y_test_l1t7,   x_test_1_l1t7,   x_test_2_l1t7 = copy.deepcopy(y_test),   copy.deepcopy(x_test_1),   copy.deepcopy(x_test_2)#l0t24567
    else:
        y_test_l1t7 = reward_output_l1t7[:100]
        x_test_1_l1t7=state_input_l1t7[:100]
        x_test_2_l1t7= action_input_l1t7[:100]
    print(np.shape(x_train_1), np.shape(x_train_2), np.shape(x_test_1), np.shape(x_test_2), np.shape(y_train), np.shape(y_test)) 
    print(np.shape(x_test_1_l1t7),   np.shape(x_test_2_l1t7),  np.shape(y_test_l1t7))

    print(np.amax(state_input), np.amin(state_input), np.amax(action_input), np.amin(action_input), np.amax(reward_output), np.amin(reward_output))
    N, C, H, W = np.shape(x_train_1)
   
    f_log = open("log_concat.txt", "w")
    if state_file is None: 
        #train
        outter_epoch_num = epoch
        inner_epoch_num = 100
        model.train()
        batch_size = 11
        os.mkdir("./saved_model_L0")
        reward_all = []
        for outter_epoch in range(outter_epoch_num + 1):
                count = 0
                running_loss = 0.0
                '''
                #load file from folder
                path_to_data = "./data/120000pts/100_level_0_type_2_novelties/"
                folder_name = [ "r_2_2", "r_3_1",  "r_5_1"] #Train: [ "r_1_1",   "r_1_3",  "r_1_4",    "r_2_2",   "r_3_1", "r_4_1", "r_5_1"]  #Test: ["r_4_2"]
                for name in folder_name:
                    state_input, action_input, reward_output = collect_data_from_folder(path_to_data+name+"/")
                    state_input, action_input, reward_output = normalization(state_input, action_input, reward_output)
                    reward_output = np.reshape(reward_output, [-1,1])
                    print(np.shape(state_input), np.shape(action_input), np.shape(reward_output))
                    print(np.amax(state_input), np.amin(state_input), np.amax(action_input), np.amin(action_input), np.amax(reward_output), np.amin(reward_output))
                    x_test_1 =  np.load("x_test_1.npy")
                    x_test_2 =  np.load("x_test_2.npy")
                    y_test   =  np.load("y_test.npy")  
                '''                
                                
                if True:          
                    '''                
                    n = len(state_input)
                    x_train_1 = state_input
                    x_train_2 = action_input
                    y_train = reward_output
                    '''
                    n = len(y_train)
                    reward_all.append(np.reshape(y_train, [n,1]))
                    count = count +1
                    print(np.shape(x_train_1), np.shape(x_train_2), np.shape(x_test_1), np.shape(x_test_2), np.shape(y_train), np.shape(y_test))
                    N, C, H, W = np.shape(x_train_1)
                    if True:
                    #for inner_epoch in range(inner_epoch_num+1):
                        #epoch = outter_epoch*10 + inner_epoch
                        for i in range(int(N/batch_size)):
                            optimizer.zero_grad()
                            output = model(torch.tensor(x_train_1[i*batch_size: (i+1)*batch_size,:,:,:]).float().cuda(), torch.tensor(x_train_2[i*batch_size: (i+1)*batch_size,:]).float().cuda())
                            loss = criterion(output, torch.tensor(y_train[i*batch_size: (i+1)*batch_size,:]).float().cuda())
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item()
                            if i%10 == 0:
                                print("loss of {} epoch  , {} index out of {} : {}".format(outter_epoch, i, N/batch_size, running_loss/float(i+1)))
                                f_log.write("loss of {} epoch  , {} index out of {} : {}".format(outter_epoch, i, N/batch_size, running_loss/float(i+1)))
                                f_log.write("\n")
                            if epoch % 100 == 0 and epoch > 0 :
                                for j in range(int(N*0.02/batch_size)):
                                    y_hat = model(torch.tensor(x_test_1[j*batch_size: (j+1)*batch_size, :,:,:]).float().cuda(),torch.tensor(x_test_2[j*batch_size: (j+1)*batch_size, :]).float().cuda())
                                    val_loss = criterion(y_hat, torch.tensor(y_test[j*batch_size: (j+1)*batch_size, :]).float().cuda()).item()
                                    print("val Loss (total epoch"+str(int(epoch))+"): "+str(val_loss))
                                    f_log.write("\n")
                                    f_log.write("val Loss (total epoch"+str(int(epoch))+"): "+str(val_loss))
                                    f_log.write("\n")
                                    y_hat  =  y_hat.cpu().detach().numpy()
                                PATH = './saved_model_L0/'+str(epoch)+'.pth'
                                torch.save(model.state_dict(), PATH)
        reward_save=np.concatenate(reward_all, axis=0)
        d1,d2 =np.shape(reward_save)
        reward_save = np.reshape(reward_save, [d1*d2,1])
        np.save("reward_train.npy", reward_save)
    else:
         y_max = np.load("reward_max.npy")
         y_min = np.load("reward_min.npy")
         print("testing model, loading model")
         model.load_state_dict(torch.load(state_file))
         # evaluate accuracy on the training data
         n = len(x_test_1)
         Y_pred = []
         mse_all = []
         for i in range(n):
             Y_pred_ = model(torch.tensor(x_test_1[i:i+1]).float().cuda(),torch.tensor(x_test_2[i:i+1]).float().cuda())
             Y_pred_ = Y_pred_.cpu().detach().numpy()
             #print( x_test_2[i:i+1], y_test[i,0], y_test[i,0],y_max[0], y_min[0])
             
             #denorm
             #print(np.shape(y_test), np.shape(Y_pred_))#(52, 1) (1, 1)
             #print(y_test[i,0], Y_pred_[0,0]) #0.14630434782608695 0.21304667
             y_test[i,0]  = y_test[i,0]*(y_max[0]-y_min[0])+y_min[0]
             Y_pred_[0,0] = Y_pred_[0,0]*(y_max[0]-y_min[0])+y_min[0] #replace
             #print(y_test[i,0], Y_pred_[0,0]) #6730.0 9800.146
             mse = np.mean( (y_test[i,0] - Y_pred_[0,0])**2)
             #print(y_test[i,0], Y_pred_[0,0])
             
             print(str(i+1)+"th "+" MSE (L0T22-25): "+str(mse)) #after denorm
             mse_all.append(mse)
             #print(Y_pred_)
             Y_pred.append(Y_pred_)
         Y_pred = np.concatenate(Y_pred, axis=0)
         mse = np.mean(mse_all) 
         rmse1 =  mse**0.5         
         print("Total MSE (L0T24567): "+str(mse))
         print("Total RMSE (L0T24567): "+str(rmse1))
         print("+++++++++++++++++++++++++++++++++++++++++\n\n\n\n")
         # Test for un-trained Level1 Type 7 data
         n = len(x_test_1_l1t7)
         Y_pred_l1t7 = []
         mse_all_l1t7 = []
         rmse_all_l1t7 = []
         for i in range(n):
             Y_pred_l1t7_ = model(torch.tensor(x_test_1_l1t7[i:i+1]).float().cuda(),torch.tensor(x_test_2_l1t7[i:i+1]).float().cuda())
             Y_pred_l1t7_ = Y_pred_l1t7_.cpu().detach().numpy()
             #denorm
             #print(np.shape(y_test_l1t7), np.shape(Y_pred_l1t7_))#(52, 1) (1, 1)
             #print( x_test_2_l1t7[i:i+1], y_test_l1t7[i,0], Y_pred_l1t7_[0,0], y_max[0], y_min[0])
             y_test_l1t7[i,0]  = y_test_l1t7[i,0]*(y_max[0]-y_min[0])+y_min[0]
             Y_pred_l1t7_[0,0] = Y_pred_l1t7_[0,0]*(y_max[0]-y_min[0])+y_min[0]                         
             mse_l1t7 = np.mean( (y_test_l1t7[i,0] - Y_pred_l1t7_[0,0])**2)
             #print( y_test_l1t7[i,0], Y_pred_l1t7_[0,0])
             print(str(i+1)+"th "+" MSE (L17): "+str(mse_l1t7))
             mse_all_l1t7.append(mse_l1t7)
             rmse_all_l1t7.append(mse_l1t7**0.5)
             #print("RMSE : "+str(mse_l1t7**0.5))
             Y_pred_l1t7.append(Y_pred_l1t7_)
         Y_pred_l1t7 = np.concatenate(Y_pred_l1t7, axis=0)
        
         mse_l1t7 = np.mean(mse_all_l1t7)    
         rmse_l1t7 = mse_l1t7**0.5
         print("Total MSE for novel data(L0T22): "+str(mse_l1t7))
         print("Total RMSE (L0T24567): "+str(rmse_l1t7))

         os.mkdir("reward_prediction_train_L0T222225_test_"+DATA)

         print(y_test.shape)
         np.save("reward_prediction_train_L0T222225_test_"+DATA + "/y_train.npy", y_train)
         np.save("reward_prediction_train_L0T222225_test_"+DATA + "/y_test.npy", y_test)
         np.save("reward_prediction_train_L0T222225_test_"+DATA + "/y_test_prediction.npy", Y_pred)
         fout = open("reward_prediction_train_L0T222225_test_"+DATA + "/output.csv", "w")
         fout.write("ground truth (l0), prediction (l0),ground truth (L0t22), prediction (L0t22) " + "\n")
         gt =[]
         pr = []
         gt_l1t7 =[]
         pr_l1t7 = []
         print(np.shape(Y_pred), np.shape(y_test), np.shape(y_test_l1t7), np.shape(Y_pred_l1t7))
         for j in range(y_test.shape[0]):
             fout.write(str(y_test[j,0])+ ", "+str(Y_pred[j,0])+" ,"+str(y_test_l1t7[j,0])+ ", "+str(Y_pred_l1t7[j,0])+"\n")
             gt.append(float(y_test[j,0]))
             pr.append(float(Y_pred[j,0]))
         #for j in range(y_test_l1t7.shape[0]):
             gt_l1t7.append(float(y_test_l1t7[j,0]))
             pr_l1t7.append(float(Y_pred_l1t7[j,0]))
         fig, ax = plt.subplots()
         ax.scatter(gt, pr, marker='.', label="trained with L0T222-225, test with L0T222-225" )
         ax.scatter(gt_l1t7, pr_l1t7, marker='*', c='red', label="trained with L0T222-225, test with "+DATA)
         ax.legend()
         ax.grid(True)
         plt.xlabel("Ground Truth")
         plt.ylabel("Prediction")
         plt.title("Only trained with L0T222-225\n"+"RMSE (L0T222-225) :"+str('%.3f'%(rmse1)+"\nRMSE ("+DATA+") :"+str('%.3f'%(rmse_l1t7))))
         plt.savefig("reward_prediction_train_L0T222225_test_"+DATA +"/"+"fig_"+DATA+".png")
         plt.close()
         fout.close()
         n_bins = 20
         plt.hist(rmse_all_l1t7, n_bins)
         plt.xlabel("RMSE")
         plt.ylabel("Counts")
         if DATA == "l0t222to225":
             plt.title("Trained with L0T222-225\n"+"RMSE (L0T222-225) :"+str('%.3f'%(rmse1))) #l0t24567
         else:
             plt.title("Only trained with L0T222-225\n"+"\nRMSE ("+DATA+") :"+str('%.3f'%(rmse_l1t7)))
             

         plt.savefig("reward_prediction_train_L0T222225_test_"+DATA+ "/"+"hist_"+DATA+".png")
         plt.close()
         print(np.shape(rmse_all_l1t7))
         np.save("reward_prediction_train_L0T222225_test_"+DATA+ "/"+"rmse_"+DATA+".npy", rmse_all_l1t7)
    f_log.close()

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(10)
    epoch = 300
    state_file ='./saved_model_L0_222_225/'+str(epoch)+'.pth'
    #state_file = None
    main(state_file, epoch) # if state_file is None: training, otherwise, eval of pretraine-model
