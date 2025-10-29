from dataset import *
from model import *
import torch
import os
from torch.utils.data import DataLoader


RANDOM_STATE = 354


PRE_PROCCESSED = True #Flag to see if we've already pre-processed data 
X_PATH = "PatternAnalysis-2025/recognition/Siamese Network s4742616/train.pt"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Loads and preproccesses datasets
    print(f"Training on Device: {device}")
    #Loads Data
    train, val = get_data(device) 
    #Sets up dataset loaders
    train_loader = DataLoader(train, batch_size = 16, shuffle = True)
    val_loader = DataLoader(val, batch_size = 16, shuffle = True)

    #load models
    network = SiameseNetwork()

    #begin training loop

    #save model, maybe both? 

