#IMPORT REQUIRED PACKAGES
import csv
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#READ IN DATA FROM CSV
def read_data(file_name):
    with open(file_name, 'r') as data_file: #Open CSV data file
        data_csv = csv.reader(data_file) #Read CSV data file to a varaible
        data_classes = next(data_csv) #Read first row of the data file into a headers variable
        data_by_class = list(zip(*[[float(time_data) for time_data in data_by_time] for data_by_time in data_csv])) #Read rest of the lines row by row to get the data organized by time then rotate the 2D array using zip to get the data organized by class then convert back to list
        data_by_class = np.asarray(data_by_class, dtype=np.float64) #Convert zipped list of data into a 2D numpy array
        data_classes = np.asarray(data_classes) #Convert list of classes into a 1D numpy array
    return data_by_class, data_classes #Return the data read and class tags from the function

#RANDOMIZE DATA
def randomize_data(data_by_class, data_classes):
    data_and_classes = list(zip(data_by_class,data_classes)) #Combine data_by_class and the data_classes into a pair tuple list
    random.shuffle(data_and_classes) #Randomize data tuples this ensures the classes are paired with the proper data.
    rand_data_by_class, rand_data_classes = zip(*data_and_classes) #Unzip the tuples into the randomized data and randomized classes
    rand_data_by_class = np.asarray(rand_data_by_class, dtype=np.float64) #Convert randomized zipped list of data into a 2D numpy array
    rand_data_classes = np.asarray(rand_data_classes) #Convert randomized list of classes into a 1D numpy array
    return rand_data_by_class, rand_data_classes #Return the randomized data by class and linked randomized class tags

#SCALE DATA WITH ROLLING MEAN
def scale_data_using_rollingMean(data_by_class, target_output_size):
    num_time_series_points = data_by_class.shape[1]
    if(num_time_series_points>=target_output_size):
        scaled_class_data = []
        for class_index, class_data in enumerate(data_by_class):
            scaled_class_data.append([])
            initial_increment = (len(class_data)+1)-target_output_size
            data_index = initial_increment
            while data_index < len(class_data)+1:
                moving_average_frame = class_data[data_index-initial_increment:data_index]
                average_of_frame = np.sum(moving_average_frame)/len(moving_average_frame)
                scaled_class_data[class_index].append(average_of_frame)
                data_index+=1
        return np.asarray(scaled_class_data, dtype=np.float64)
    else:
        return np.asarray(data_by_class, dtype=np.float64)

#SCALE DATA WITH DATA REMOVAL
def scale_data_using_data_removal(data_by_class, increment):
    scaled_class_data = []
    for class_index, class_data in enumerate(data_by_class):
        scaled_class_data.append([])
        data_index = -1
        while data_index < len(class_data)-increment:
            data_index+=increment
            scaled_class_data[class_index].append(class_data[data_index])
            
    return np.asarray(scaled_class_data, dtype=np.float64)

#CROP DATA
def crop_data(data_by_class,cutoff_value):
    return np.asarray([class_data[:cutoff_value] for class_data in data_by_class], dtype=np.float64)

#NORMALIZE DATA
def normalize_data(data_by_class,normalize_range):
    scaler = MinMaxScaler(feature_range=normalize_range)
    scaler.fit(data_by_class)
    return np.asarray(scaler.transform(data_by_class), dtype=np.float64)

#PREPARE CLASS TAGS STRING TO FLOAT
def prepare_tags(data_classes):
    converted_data_classes = []
    for data_class in data_classes:
        split_text = data_class.split("-")
        converted_data_classes.append([float(text) for text in split_text])
    return(np.asarray(converted_data_classes, dtype=np.float64))