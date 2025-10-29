import torch
import numpy as np
import pandas as pd
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#Dataset sources
Y_DATA = "PatternAnalysis-2025/recognition/Siamese Network s4742616/data/ISIC_2020_Training_GroundTruth.csv" #Y data source
X_DATA = "PatternAnalysis-2025/recognition/Siamese Network s4742616/data/train/" #X data source

#Temp File Names
FILENAME_X = "PatternAnalysis-2025/recognition/Siamese Network s4742616/train2.pt"

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
    transform = transforms.Compose([
        transforms.Resize((DIMENSIONS, DIMENSIONS)),
        transforms.ToTensor(),
    ])

    images = []

    for filename in tqdm(os.listdir(X_DATA), desc = "Processing Images"):
        if filename.lower().endswith('.jpg'): #It will
            image_path = os.path.join(X_DATA, filename)
            image = Image.open(image_path).convert("RGB") #Keeps colours
            image = transform(image) #Convert to tensor, resize
            images.append(image)
    
    X = torch.stack(images) #Converts to single tensor

    torch.save(X, FILENAME_X)
    

    return FILENAME_X


def preproccess_Y(device = "cpu", Y_source = Y_DATA):
    """ Preproccesses Y Data from csv to pytorch tensors
        Returns tensor containing ground truth values for training set
    """

    df = pd.read_csv(Y_DATA)
    df = df[['target']] #Just gets the ground truth values, leaves rest out

    y = torch.from_numpy(df.values).to(device) #Converts to tensor, puts to device

    return y

def train_test_valid_split(X, Y):
    indices = list(range(len(X))) #indices of X data
    #Split 70% of the data into training set
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(indices, Y, test_size = 0.3, stratify = Y)
    #Split 20% of overall data into test set, 10% into validation
    val_idx, test_idx, val_labels, test_labels = train_test_split(temp_idx, temp_labels, test_size = 0.66, stratify = temp_labels)


    x_train = X[train_idx]
    y_train = train_labels

    x_val = X[val_idx]
    y_val = val_labels

    x_test = X[test_idx]
    y_test = test_labels

    return x_train, y_train, x_val, y_val, x_test, y_test
