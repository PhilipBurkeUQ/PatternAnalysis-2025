import torch
import numpy as np
import pandas as pd
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random

#Dataset sources
Y_DATA = "PatternAnalysis-2025/recognition/Siamese Network s4742616/data/ISIC_2020_Training_GroundTruth.csv" #Y data source
X_DATA = "PatternAnalysis-2025/recognition/Siamese Network s4742616/data/train/" #X data source

#Temp File Names
FILENAME_X = "PatternAnalysis-2025/recognition/Siamese Network s4742616/data.pt"
TRAIN_DATA = "PatternAnalysis-2025/recognition/Siamese Network s4742616/trainDataset.pt"
#Some default parameters
DIMENSIONS = 256 #Dimensions of resized image
RANDOM_STATE = 354

PREPROCESS_NEEDED = False #Flag to determine whether to preprocess images or not (i.e should I load from file or not)
MAKE_DATASET = True
def get_data(device = "cpu"):
    #X = preprocess_X(device)
    Y = preprocess_Y(device)
    train_idx, val_idx, test_idx = train_test_valid_split(X_DATA, Y)

    #Convert to Dataset class
    if MAKE_DATASET or not(os.path.exists(TRAIN_DATA)):
        train_dataset = SiameseDataset(X_DATA,Y,train_idx)
        torch.save(train_dataset, TRAIN_DATA)
    else:
        train_dataset = torch.load(TRAIN_DATA, weights_only = False)

    return train_dataset
    



def preprocess_X(device = "cpu", X_source = X_DATA, newSize = DIMENSIONS):
    """ Preprocesses X data from images to pytorch tensors
        Preprocessing Steps:
        1.Convert to tensor
        2.Resize to square (dim * dim)
        3.Data augmentation (Possibly)
        Returns filename of tensor saved to file """
    if not PREPROCESS_NEEDED:
        return torch.load(FILENAME_X, weights_only = False) #If we aren't pre-processing load from file
    
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


def preprocess_Y(device = "cpu", Y_source = Y_DATA):
    """ Preproccesses Y Data from csv to pytorch tensors
        Returns tensor containing ground truth values for training set
    """

    df = pd.read_csv(Y_DATA)
    df = df[['target']] #Just gets the ground truth values, leaves rest out

    y = torch.from_numpy(df.values).to(device) #Converts to tensor, puts to device

    return y

def train_test_valid_split(X_DATA, Y):
    """
    Returns the indices of the datapoints in the train/validation/test"""
    indices = list(range(len(os.listdir(X_DATA)))) #indices of X data
    #Split 70% of the data into training set
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(indices, Y, test_size = 0.3, stratify = Y)
    #Split 20% of overall data into test set, 10% into validation
    val_idx, test_idx, val_labels, test_labels = train_test_split(temp_idx, temp_labels, test_size = 0.66, stratify = temp_labels)

    return train_idx, val_idx, test_idx

class SiameseDataset(torch.utils.data.Dataset):
    #Custom dataset for pairs of data
    #To be used with data loader
    def __init__(self, x_data, y_data, indices):
        import pandas as pd
        from pathlib import Path
        self.x_data = Path(x_data)
        self.y_data = y_data
        self.indices = indices
        torch.manual_seed(RANDOM_STATE)
        self.transform = transforms.Compose([
            transforms.Resize((DIMENSIONS, DIMENSIONS)),
            transforms.ToTensor(),
            ])
        self.filenames = [f.name for f in self.x_data.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg']]
        assert len(self.filenames) == len(y_data), f"Number of images ({len(self.filenames)}) does not match number of labels ({len(y_data)})"
        self._pair_data()

    def _pair_data(self):
        benign_idx = [i for i in self.indices if self.y_data[i] == 0]
        malig_idx =  [i for i in self.indices if self.y_data[i] == 1]

        self.pos_pairs = []
        for idx_list in [benign_idx, malig_idx]:
            for _ in range((5000)):
                i, j = random.sample(idx_list, 2)
                self.pos_pairs.append((i,j))
        self.pos_labels = [1] *len(self.pos_pairs)

        # Sample negative pairs
        num_negatives = 5000
        self.neg_pairs = []
        for _ in range(num_negatives):
            i = random.choice(benign_idx)
            j = random.choice(malig_idx)
            self.neg_pairs.append((i, j))
        self.neg_labels = [0] * len(self.neg_pairs)
        
        # Combine and shuffle
        self.pairs = self.pos_pairs + self.neg_pairs
        self.pair_labels = self.pos_labels + self.neg_labels
        combined = list(zip(self.pairs, self.pair_labels))
        random.shuffle(combined)
        self.pairs, self.pair_labels = zip(*combined)


    def __len__(self):
        return len(self.pairs)
    
    def _load_image(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.x_data, filename)
        img = Image.open(image_path).convert("RGB")
        return self.transform(img)


    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        img1, img2 = self._load_image(i), self._load_image(j)
        label = torch.tensor(self.pair_labels[idx], dtype=torch.float32)
        return img1, img2, label
