import utils
import matplotlib.pyplot as plt

data_by_paramters, data_parameters = utils.read_data("APData.csv") #Read in CSV data into two lists. First contains the data organized by different action potential firign paramters. Second contains a list of the firing paramters used.
cropped_data_by_paramters = utils.crop_data(data_by_paramters,1000) #Crop data of constant voltage
scaled_data_by_parameters = utils.scale_data_using_data_removal(cropped_data_by_paramters,2) #Shrink the random data to be smaller
for data, label in zip(scaled_data_by_parameters, data_parameters):
   plt.plot(data)
plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
plt.show()