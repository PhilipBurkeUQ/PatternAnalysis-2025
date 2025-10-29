from dataset import *
import torch
import os



RANDOM_STATE = 354


PRE_PROCCESSED = True #Flag to see if we've already pre-processed data 
X_PATH = "PatternAnalysis-2025/recognition/Siamese Network s4742616/train.pt"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Loads and preproccesses datasets
    print(f"Training on Device: {device}")
    Y = preproccess_Y(device)
    if PRE_PROCCESSED and os.path.exists(X_PATH):
        X_name = X_PATH
    else:
        X_name = preprocess_X(device)
    X = torch.load(X_name)

    #Splits data into train, test and validation splits
    x_train, y_train, x_val, y_val, x_test, y_test = train_test_valid_split(X,Y)

    x_test, y_test = 0,0 #Don't need them for training



    #load models

    #begin training loop

    #save model, maybe both? 

