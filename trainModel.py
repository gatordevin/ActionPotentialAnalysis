#IMPORT REQUIRED PACKAGES
import utils
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import models
import os

#DATA PREPROCESSING
data_by_paramters, data_parameters = utils.read_data("APData.csv") #Read in CSV data into two lists. First contains the data organized by different action potential firign paramters. Second contains a list of the firing paramters used.
cropped_data_by_paramters = utils.crop_data(data_by_paramters,1000) #Crop data of constant voltage
rand_data_by_parameters, rand_data_parameters = utils.randomize_data(cropped_data_by_paramters, data_parameters) #Randomize the dataset in a way that keeps the parameters and firing data associated with those paramters at the same index
scaled_rand_data_by_parameters = utils.scale_data_using_data_removal(rand_data_by_parameters,2) #Shrink the random data to be smaller
normalized_data = utils.normalize_data(scaled_rand_data_by_parameters,(-1,1)) #Normalize the data to values between -1 and 1 so that high values dont prevent the model from training properly
parameters_split = utils.prepare_tags(rand_data_parameters) #Split string tags into numbers
tensor_data = torch.from_numpy(normalized_data).float() #Conevrt Normalized and preprocessed data into a tensor that can be fed into the network
tensor_parameters = torch.from_numpy(parameters_split).float() #Convert paramters into a tensor that can be fed into the network

#CREATE MODEL
time_series_dims = normalized_data[0].ndim #Get size of time series data recordings in this case it is only voltage
LSTM_hidden_layer = 400 #How many LSTMS are in this layer
num_of_parameters = parameters_split[0].size #Get number of parameters that need to be predicted
learning_rate = 0.000001 #Learning rate for model
epochs=4000 #How many times will the model go through the data
model_save_increments = 100 #How often will loss be measured and will the model be saved

model_type = "models/SingleLSTMLayer"
path = model_type + "_" + str(LSTM_hidden_layer) + "_" + str(learning_rate) + "/"
if not os.path.isdir(path):
    os.mkdir(path)
model = models.SingleLSTMLayer(input_size=time_series_dims, hidden_layer_size=LSTM_hidden_layer, output_size=num_of_parameters)
model.cuda()
loss_function = nn.MSELoss().cuda() #Define the type of loss function the model will use for training
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #Create the optimizer object
loss_graph = []

for i in range(epochs):
    avg_loss = 0
    class_num = 0
    for seq, labels in zip(tensor_data,tensor_parameters):
        class_num+=1

        model.zero_grad()
        y_pred = model(seq.cuda())
        single_loss = loss_function(y_pred, labels.cuda())
        single_loss.backward()
        optimizer.step()

        avg_loss+=single_loss.item()
    loss_graph.append(avg_loss/class_num)
    if i%model_save_increments == 0:
        torch.save(model.state_dict(), "/home/techgarage/ActionPotentialAnalysis/"+path+"/APModel"+str(i)+".pth")
        print(f'epoch: {i:3} loss: {avg_loss/class_num:10.8f}')

torch.save(model.state_dict(), f"/home/techgarage/ActionPotentialAnalysis/"+path+"/APModel"+str(epochs)+".pth") #Save model at end of training
plt.plot(loss_graph)
plt.show()