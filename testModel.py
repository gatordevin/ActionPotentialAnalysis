import utils
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import models
import math
#DATA PREPROCESSING
data_by_paramters, data_parameters = utils.read_data("validationData.csv") #Read in CSV data into two lists. First contains the data organized by different action potential firign paramters. Second contains a list of the firing paramters used.
cropped_data_by_paramters = utils.crop_data(data_by_paramters,1000) #Crop data of constant voltage
#rand_data_by_parameters, rand_data_parameters = utils.randomize_data(cropped_data_by_paramters, data_parameters) #Randomize the dataset in a way that keeps the parameters and firing data associated with those paramters at the same index
scaled_rand_data_by_parameters = utils.scale_data_using_data_removal(cropped_data_by_paramters,2) #Shrink the random data to be smaller
normalized_data = utils.normalize_data(scaled_rand_data_by_parameters,(-1,1)) #Normalize the data to values between -1 and 1 so that high values dont prevent the model from training properly
parameters_split = utils.prepare_tags(data_parameters) #Split string tags into numbers
tensor_data = torch.from_numpy(normalized_data).float() #Conevrt Normalized and preprocessed data into a tensor that can be fed into the network
tensor_parameters = torch.from_numpy(parameters_split).float() #Convert paramters into a tensor that can be fed into the network

device = torch.device('cuda')
time_series_dims = normalized_data[0].ndim #Get size of time series data recordings in this case it is only voltage
LSTM_hidden_layer = 400 #How many LSTMS are in this layer
num_of_parameters = parameters_split[0].size #Get number of parameters that need to be predicted
learning_rate = 0.00001 #Learning rate for model
epochs=2000 #How many times will the model go through the data
model_save_increments = 100 #How often will loss be measured and will the model be saved
time_series_length = len(normalized_data[0])
hidden_layer_size = 200
num_of_layers = 1

#model = ActionPotentialParamterPredictor(input_size=time_series_dims, hidden_layer_size=LSTM_hidden_layer, output_size=num_of_parameters)
#model = models.FullyConnectedModel(input_size=time_series_length,hidden_size=hidden_layer_size,num_of_hidden_layers=num_of_layers,output_size=num_of_parameters)
model = models.SingleLSTMLayer(input_size=time_series_dims, hidden_layer_size=LSTM_hidden_layer, output_size=2)
model.to(device)
sodiumErrors = []
potassiumErrors = []
for epoch in range(4000):
    if(epoch%100 == 0):
        print(epoch)
        model.load_state_dict(torch.load("/home/techgarage/ActionPotentialAnalysis/models/SingleLSTMLayer_400_1e-06/APModel"+str(epoch)+".pth"))
        model.eval()

        sodium_pred = []
        potassium_pred = []

        sodium_truth = []
        potassium_truth = []

        for paramter, data in zip(tensor_parameters,tensor_data):
            with torch.no_grad():
                pred = model(data.to(device))
                #print(pred)
            sodium_pred.append(pred.cpu().numpy()[0])
            potassium_pred.append(pred.cpu().numpy()[1])
            sodium_truth.append(paramter.cpu().numpy()[0])
            potassium_truth.append(paramter.cpu().numpy()[1])

        averageError = 0
        count = 0
        for pred, truth in zip(sodium_pred, sodium_truth):
            if truth != 0:
                count += 1
                error = abs(pred - truth)/truth
                averageError += error
        sodiumAverageError = averageError/(count+1)*100

        averageError = 0
        count = 0
        for pred, truth in zip(potassium_pred, potassium_truth):
            if truth != 0:
                count += 1
                error = abs(pred - truth)/truth
                averageError += error
        potassiumAverageError = averageError/(count+1)*100
        print(potassiumAverageError)
        print(sodiumAverageError)
        sodiumErrors.append(sodiumAverageError)
        potassiumErrors.append(potassiumAverageError)

fig, axs = plt.subplots(2)
fig.suptitle('Accuracy over training')
axs[0].plot(sodiumErrors)
axs[0].set(xlabel='num of epochs', ylabel='Sodium Pred Error (%)')
axs[1].plot(potassiumErrors)
axs[1].set(xlabel='num of epochs', ylabel='Potassium Pred Error (%)')
plt.show()

# fig, axs = plt.subplots(2)
# fig.suptitle('Vertically stacked subplots')
# axs[0].plot(sodium_pred, label="sodium_pred")
# axs[1].plot(potassium_pred, label="potassium_pred")
# axs[0].plot(sodium_truth, label="sodium_truth")
# axs[1].plot(potassium_truth, label="potassium_truth")
# axs[0].legend(loc="upper left")
# axs[1].legend(loc="upper left")
# plt.show()