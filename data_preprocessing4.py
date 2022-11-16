import numpy as np
import os, sys

#for i in range(108):
#    os.system("python cnn.py "+str(i))

y = []    
epoch = sys.argv[1] 
for i in range(46):
    print(i)
    y.append(np.load("y_hat2_"+str(i)+".npy"))
Y_pred = np.concatenate(y, axis = 0)
print(np.shape(Y_pred))
np.save("Y_pred3_"+str(epoch)+".npy", Y_pred)