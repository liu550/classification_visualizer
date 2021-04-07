import tensorflow as tf
from tensorflow.keras import layers
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys

#global variables
coordinates_to_data_points = {}

class data_point():
    def __init__(self, features, label, prediction, match):
        self.features = features
        self.label = label
        self.prediction = prediction
        self.match = match
    

def append_one_hot_to_class(one_hot, classification, dp):
    i = 0
    classification_list = list(classification)
    for i in range(len(one_hot)):
        if one_hot[i] == 1:
            classification[classification_list[i]].append(dp)

def calculate_size(size_of_data):
    i = 1
    for i in range(sys.maxsize):
        if i*i*(1+5**0.5)/2 > size_of_data:
            return i*(1+5**0.5)/2, i

def prediction_is_correct(y, y_prediction):
    return abs(y-y_prediction) <= 0.5


def prediction_to_one_hot(y, y_prediction):
    one_hot = []
    i = 0
    match = True

    for i in range(len(y_prediction)):
       
        if prediction_is_correct(y[i], y_prediction[i]) == False:
            match = False
        if y_prediction[i] > 0.5:
            one_hot.append(1)
        else:
            one_hot.append(0)
    return one_hot, match



def on_click(event):
    x_coordinate = round(event.xdata)
    y_coordinate = round(event.ydata)
    #round the coordinates to the closet integer
    stringified_coordinates = str(x_coordinate) + ',' + str(y_coordinate)
    dp = coordinates_to_data_points[stringified_coordinates]
    print()
    print(f'Inputs are: {dp.features}')
    print(f'Label is: {dp.label}')
    print(f'Prediction made by the model is: {dp.prediction}')
    print('Correct') if dp.match else print('Wrong')

def classification_visualizer(model, x, y, classes):
    
    y_predictions = model.predict(x)

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    #build classification dictionary
    classification = {}
    for category in classes:
        classification.update({category: []})

    #convert y_prediction to one-hot
    i = 0

    classification_list = list(classification)

    #binary
    if len(y_predictions[0]) == 1:
        for i in range(len(y_predictions)):
            value = y_predictions[i]
            if prediction_is_correct(y[i], y_predictions[i]):
                dp = data_point(x[i], y[i], y_predictions[i], True)
            else:
                dp = data_point(x[i], y[i], y_predictions[i], False)
            index = 1 if y_predictions[i] > 0.5 else 0
            classification[classification_list[index]].append(dp)
            
            
    #multiclass
    else:
        for i in range(len(y_predictions)):
            y_prediction = y_predictions[i]
            print(y_prediction)
            print(y[i])
            one_hot, match = prediction_to_one_hot(y[i], y_prediction)
            #initialize a data point object
            dp = data_point(x[i], y[i], y_predictions[i], match)

            #append one_hot to the corresponding class list
            append_one_hot_to_class(one_hot, classification, dp)

        if i == 0:
            print('for debugging:')
            print(f'one_hot looks like {one_hot}')
            print(f'dp looks like {dp}')
            print(f'classification looks like {classification}')
    
    #size of the plot
    x_max, y_max = calculate_size(len(y))

    #assign coordinates to each data point object and create a dictionary that associates the two
    y_coordinate = 1
    x_coordinate = 1
    i = 0

    fig, ax = plt.subplots()

    for i in range(len(classification)):

        correct_predictions_x = []
        correct_predictions_y = []
        wrong_predictions_x = []
        wrong_predictions_y = []

        print(classification_list[i])
        for dp in classification[classification_list[i]]:
            if x_coordinate > x_max:
                x_coordinate = 1
                y_coordinate += 1
            stringified_coordinates = str(x_coordinate) + ',' + str(y_coordinate)
            coordinates_to_data_points.update({stringified_coordinates: dp})
            #plot
            '''
            print(f'label is {dp.label}. type of label is {type(dp.label)}')
            print(f'prediction is {dp.prediction}. type is {type(dp.prediction)}')
            '''

            '''
            marker = 'o' if dp.match else 'x'
            ax.plot(x_coordinate, y_coordinate, marker=marker, label=classification_list[i], color=colors[i])
            '''
            if dp.match:
                correct_predictions_x.append(x_coordinate)
                correct_predictions_y.append(y_coordinate)
            else:
                wrong_predictions_x.append(x_coordinate)
                wrong_predictions_y.append(y_coordinate)
            x_coordinate += 1

        ax.scatter(correct_predictions_x, correct_predictions_y, marker='o', label=classification_list[i], color=colors[i])
        ax.scatter(wrong_predictions_x, wrong_predictions_y, marker='x', label=classification_list[i], color=colors[i])
        ax.legend(bbox_to_anchor=(0.85, 1.2), loc='upper left', fontsize='x-small')

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()







