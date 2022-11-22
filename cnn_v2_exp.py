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
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = torch.cat((x,u),1)
        x = self.fc(x)

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

    #print(model)
    #trouble shooting
    #model(torch.randn(6, 13, 300, 600).cuda(), torch.randn(6,5).cuda()).float().cuda()
    #load data from scratch
    """
    state_input, action_input, reward_output = collect_data("./soo_novelty_detection/data/non-novelty/")
    state_input, action_input, reward_output = normalization(state_input, action_input, reward_output)
    #state_input, action_input, reward_output = normalization_L1T7(state_input, action_input, reward_output)
    print(np.shape(state_input), np.shape(action_input), np.shape(reward_output))
    print(np.amax(state_input), np.amin(state_input), np.amax(action_input), np.amin(action_input), np.amax(reward_output), np.amin(reward_output))
    exit()
   """
    path_in = "./soo_novelty_detection/input_data/"
    state_input, action_input, reward_output = np.load(path_in+"state.npy"), np.load(path_in+"action.npy"), np.load(path_in+"reward.npy")
    n = len(state_input)
    x_train_1, x_test_1 = state_input[:int(n*0.9)], state_input[int(n*0.9):]
    x_train_2, x_test_2 = action_input[:int(n*0.9)], action_input[int(n*0.9):]
    y_train, y_test = reward_output[:int(n*0.9)] , reward_output[int(n*0.9):]
    print(np.shape(x_train_1), np.shape(x_train_2), np.shape(x_test_1), np.shape(x_test_2), np.shape(y_train), np.shape(y_test))
    print(np.amax(state_input), np.amin(state_input), np.amax(action_input), np.amin(action_input), np.amax(reward_output), np.amin(reward_output))
    N, C, H, W = np.shape(x_train_1)    
    f_log = open("./soo_novelty_detection/log_concat.txt", "w")  
    if state_file is None: 
        #train
        outter_epoch_num = epoch
        model.train()
        batch_size = 10
        os.mkdir("./soo_novelty_detection/saved_model_non-novelty/")
        reward_all = []
        for outter_epoch in range(outter_epoch_num + 1):
                count = 0
                running_loss = 0.0     
                if True:          
                    n = len(y_train)
                    reward_all.append(np.reshape(y_train, [n,1]))
                    count = count +1
                    print(np.shape(x_train_1), np.shape(x_train_2), np.shape(x_test_1), np.shape(x_test_2), np.shape(y_train), np.shape(y_test))
                    N, C, H, W = np.shape(x_train_1)
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
                    if outter_epoch % 100 == 0 and outter_epoch > 0 :
                        for j in range(int(N*0.02/batch_size)):
                            y_hat = model(torch.tensor(x_test_1[j*batch_size: (j+1)*batch_size, :,:,:]).float().cuda(),torch.tensor(x_test_2[j*batch_size: (j+1)*batch_size, :]).float().cuda())
                            val_loss = criterion(y_hat, torch.tensor(y_test[j*batch_size: (j+1)*batch_size, :]).float().cuda()).item()
                            print("val Loss (total epoch"+str(int(outter_epoch))+"): "+str(val_loss))
                            f_log.write("\n")
                            f_log.write("val Loss (total epoch"+str(int(outter_epoch))+"): "+str(val_loss))
                            f_log.write("\n")
                            y_hat  =  y_hat.cpu().detach().numpy()
                        PATH = './soo_novelty_detection/saved_model_non-novelty/'+str(outter_epoch)+'.pth'
                        torch.save(model.state_dict(), PATH)
        reward_save=np.concatenate(reward_all, axis=0)
        d1,d2 =np.shape(reward_save)
        reward_save = np.reshape(reward_save, [d1*d2,1])
        np.save("./soo_novelty_detection/input_data/reward_train.npy", reward_save)
        f_log.close()
    else:
         n_test_size =81#149
         name_output_dir = "test"
         data_collecting_mode = 0 #True
         #(0) prepare test dataset
         #path of test dataset to load
         p = "./soo_novelty_detection/data/"
         # test only maximum 10 test sets
         
         path_test = [p+"novelty_level_22/0_40_novelty_level_22_type1_non-novelty_type11130",
                      p+"novelty_level_22/0_40_novelty_level_22_type2_non-novelty_type11130"]
         label = []
         for file_name in path_test:
             label_name = ""
             for char in  file_name.split("/")[-1].split("_")[3:6]:
                 label_name += char
             print(label_name)
             label.append(label_name)
         print(label)
         if data_collecting_mode:
             for i in range(len(path_test)):
                 print(i, path_test[i])
                 state_input, action_input, reward_output = collect_data_from_folder(path_test[i])
                 state_input, action_input, reward_output = normalization_L0(state_input, action_input, reward_output)
                 np.save(path_test[i]+"/state_"+label[i]+".npy", state_input)
                 np.save(path_test[i]+"/action_"+label[i]+".npy", action_input)
                 np.save(path_test[i]+"/reward_"+label[i]+".npy", reward_output)
             exit()             
         
         #(1) load data, and model prediction
         # L0: x_test1, x_test2, y_test, Y_pred
         x_test_1 = x_test_1[:n_test_size]; x_test_2 = x_test_2[:n_test_size]; y_test = y_test[:n_test_size]
         model.load_state_dict(torch.load(state_file))
         Y_pred=[]
         for i in range(len(x_test_1)):
             print(i)
             Y_pred_in = model(torch.tensor(x_test_1[i:i+1,:,:,:]).float().cuda(),torch.tensor(x_test_2[i:i+1,:]).float().cuda())
             Y_pred_in  =  Y_pred_in.cpu().detach().numpy()
             Y_pred.append(Y_pred_in)
         Y_pred = np.concatenate(Y_pred,0)
         print(np.shape(Y_pred),np.shape(y_test))
         np.save("./soo_novelty_detection/prediction_values.npy", Y_pred)
         np.save("./soo_novelty_detection/ground_truth.npy", y_test)
         mse = mean_squared_error(y_test , Y_pred)
         error = np.abs(y_test - Y_pred)
         print("Total MSE (L0): "+str(mse))
         
         #(2) Load all data to test and predict results
         #path of test dataset to load
         p = "./soo_novelty_detection/data/"
         state_input_test = []; action_input_test = []; reward_output_test = []
         for i in range(len(label)):

             state_input = np.load(path_test[i]+"/state_"+label[i]+".npy")
             action_input= np.load(path_test[i]+"/action_"+label_name+".npy")
             reward_output=np.load(path_test[i]+"/reward_"+label_name+".npy")      
             state_input_test.append(state_input)
             action_input_test.append(action_input)
             reward_output_test.append(reward_output)
         Y_pred_test=[]; mse_test=[]; error_test=[]
         print(len(state_input_test), label)
         for j in range(len(state_input_test)):
             x_test1 = state_input_test[j][:n_test_size]; x_test2=action_input_test[j][:n_test_size]; y_test2 = reward_output_test[j][:n_test_size]
             Y_pred_=[]
             for i in range(len(x_test_1)):
                 Y_pred_in = model(torch.tensor(x_test1[i:i+1,:,:,:]).float().cuda(),torch.tensor(x_test2[i:i+1,:]).float().cuda())
                 Y_pred_in  =  Y_pred_in.cpu().detach().numpy()
                 Y_pred_.append(Y_pred_in)
             Y_pred_ = np.concatenate(Y_pred_,0)
             Y_pred_test.append(Y_pred_)
             print(np.shape(Y_pred_),np.shape(y_test2))
             mse = mean_squared_error(y_test2 , Y_pred_) 
             print("Total MSE: "+label[j]+" "+str(mse))  
             mse_test.append(mse)
             error = np.abs(y_test2 - Y_pred_)
             error_test.append(error)        


         
         #(3)plot: scatter overlaid together and error distribution      
         y_max = np.load("./soo_novelty_detection/input_data/reward_max.npy")
         y_min = np.load("./soo_novelty_detection/input_data/reward_min.npy")
         os.mkdir("./soo_novelty_detection/"+name_output_dir)
         #(3-1) scatter for test L0
         fout = open("./soo_novelty_detection/"+name_output_dir + "/output_L0.csv", "w")
         fout.write("ground truth, prediction" + "\n")
         fig, ax = plt.subplots()
         gt =[];pr = []
         m_l =  [".","D",",","v","^","<",">","+","s","p","X","d"]
         c_l = ["k","yellow", "green","lime","darkred","red","lightcoral","blue","purple","orangered","navy", "cyan"] #11
         for j in range(y_test.shape[0]):
             fout.write(str(y_test[j,0]*(y_max[0]-y_min[0])+y_min[0])+ ", "+str(Y_pred[j,0]*(y_max[0]-y_min[0])+y_min[0])+"\n")
             gt.append(float(y_test[j,0]*(y_max[0]-y_min[0])+y_min[0]))
             pr.append(float(Y_pred[j,0]*(y_max[0]-y_min[0])+y_min[0]))
         error_denorm = np.mean(error)*(y_max[0]-y_min[0])+y_min[0]
         ax.scatter(gt, pr, marker="o", c=c_l[0], label="level0:"+str(round(error_denorm,2)) )
         fout.close()
         # (3-2) scatter for other levels in test folder
         for i in range(len(state_input_test)):
             fout = open("./soo_novelty_detection/"+name_output_dir + "/output_"+label[i]+".csv", "w")
             fout.write("ground truth, prediction" + "\n")
             gt=[];pr=[]
             x_test1 = state_input_test[i][:n_test_size]; x_test2=action_input_test[i][:n_test_size]; y_test2 = reward_output_test[i][:n_test_size]
             Y_pred_ =Y_pred_test[i]
             for j in range(y_test2.shape[0]):
                 fout.write(str(y_test2[j,0]*(y_max[0]-y_min[0])+y_min[0])+ ", "+str(Y_pred_[j]*(y_max[0]-y_min[0])+y_min[0])+"\n")
                 gt.append(float(y_test2[j,0]*(y_max[0]-y_min[0])+y_min[0]))
                 pr.append(float(Y_pred_[j]*(y_max[0]-y_min[0])+y_min[0]))
             error_denorm = np.mean(error[i])*(y_max[0]-y_min[0])+y_min[0]
             ax.scatter(gt, pr, marker=m_l[i+1], c=c_l[i+1], label=label[i]+":"+str(round(error_denorm,2)) )
             fout.close()
             
         # Shrink current axis by 20%
         box = ax.get_position()
         ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

         # Put a legend to the right of the current axis
         ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
         #ax.legend()
         ax.grid(True)
         plt.xlabel("Ground Truth")
         plt.ylabel("Prediction")
         plt.title("Testing results: RMSE (model trained with level 0)")
         plt.savefig("./soo_novelty_detection/"+name_output_dir + "/scatter_plot.png")
         plt.close()
         #(3-1) error histogram for testing with L0
         n_bins = 20
         plt.hist(error, n_bins)
         plt.xlabel("RMSE")
         plt.ylabel("Counts")
         plt.title("RMSE for testing with level0 , RMSE: "+str(round(np.mean(error)*(y_max[0]-y_min[0])+y_min[0],2)))
         plt.savefig("./soo_novelty_detection/"+name_output_dir + "/error_hist_level0.png")
         plt.close()
         #(3-1) error histogram for testing with different levels and types
         for i in range(len(error_test)):
             plt.hist(error[i], n_bins)
             plt.xlabel("RMSE")
             plt.ylabel("Counts")
             plt.title("RMSE for testing with "+label[i]+" , RMSE: "+str(round(np.mean(error[i])*(y_max[0]-y_min[0])+y_min[0],2)))
             plt.savefig("./soo_novelty_detection/"+name_output_dir + "/error_hist_"+label[i]+".png")
             plt.close()
       



if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(10)
    epoch = 300
    state_file ='./soo_novelty_detection/saved_model_non-novelty/'+str(epoch)+'.pth'
    #state_file = None
    main(state_file, epoch) # if state_file is None: training, otherwise, eval of pretraine-model
