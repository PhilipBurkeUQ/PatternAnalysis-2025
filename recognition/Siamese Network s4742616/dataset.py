import torch
import numpy as np
import pandas as pd
import os


#Dataset sources
Y_DATA = "data\ISIC_2020_Training_GroundTruth.csv" #Y data source
X_DATA = "data\train" #X data source

#Temp File Names
FILENAME_X = "train.pt"
#FILENAME_Y = "train.py"

#Some default parameters
DIMENSIONS = 256 #Dimensions of resized image
RANDOM_STATE = 354

def preprocess_X(device = "cpu", X_source = X_DATA, newSize = DIMENSIONS):
    """ Preprocesses X data from images to pytorch tensors
        Preprocessing Steps:
        1.Convert to tensor
        2.Resize to square (dim * dim)
        3.Data augmentation (Possibly)
        Returns filename of tensor saved to file """
    
    

    return FILENAME_X


def preproccess_Y(device = "cpu", Y_source = Y_DATA):
    """ Preproccesses Y Data from csv to pytorch tensors
        Returns tensor containing ground truth values for training set
    """

    df = pd.read_csv(Y_DATA)
    df = df[['target']] #Just gets the ground truth values, leaves rest out

    y = torch.from_numpy(df.values).to(device) #Converts to tensor, puts to device

    return y

