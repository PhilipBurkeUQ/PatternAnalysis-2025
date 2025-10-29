from dataset import *
import torch
import os

RANDOM_STATE = 354


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cude_is_available() else "cpu")
    #Loads and preproccesses datasets
    print(f"Training on Device: {device}")
    Y = preproccess_Y(device)
    print(Y[1]) #Test, please remove!
    X_name = preprocess_X(device, type)
    print(X_name) #Test, please remove!
    X = torch.load(X_name)

    #Splits data into train and test splits

