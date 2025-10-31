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
#Some default parameters
DIMENSIONS = 256 #Dimensions of resized image
RANDOM_STATE = 354

NUM_TRAIN_SAMPLES = 5000 #Parameter to control the number of pairs created

PREPROCESS_NEEDED = False #Flag to determine whether to preprocess images or not (i.e should I load from file or not)
MAKE_DATASET = True

def get_data(device = "cpu"):
    """Gets custom datasets for dataloaders for trainining data"""
    #Loads Y values to memory
    Y = preprocess_Y(device)

    #Get train, test split
    train_idx, val_idx, _ = train_test_valid_split(X_DATA, Y)

    #Convert to Dataset class
    train_dataset_embed = EmbeddingDataset(X_DATA,Y,train_idx)
    train_dataset_classify = ClassifyDataset(X_DATA, Y, train_idx)

    return train_dataset_embed, train_dataset_classify
    

def get_test_val_data(device = "cpu"):
    """Gets custom dataset for test and validation sets"""
    #Loads Y and creates train/test split
    Y = preprocess_Y(device)
    _, val_idx, test_idx = train_test_valid_split(X_DATA, Y)

    #Creates custom datasets
    val_data = ClassifyDataset(X_DATA, Y, val_idx)
    test_data = ClassifyDataset(X_DATA, Y, test_idx)

    return val_data, test_data


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

class EmbeddingDataset(torch.utils.data.Dataset):
    """Custom dataset for pairs of data
    To be used with data loader"""
    def __init__(self, x_data, y_data, indices):
        import pandas as pd
        from pathlib import Path

        #Initialise values
        self.x_data = Path(x_data)
        self.y_data = y_data
        self.indices = indices

        #Set some default parameters
        torch.manual_seed(RANDOM_STATE)
        self.transform = transforms.Compose([
            transforms.Resize((DIMENSIONS, DIMENSIONS)),
            transforms.ToTensor(),
            ])
        #Load filenames from image directory
        self.filenames = [f.name for f in self.x_data.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg']]

        #Set up pairs of data
        self._pair_data()

    def _pair_data(self):
        benign_idx = [i for i in self.indices if self.y_data[i] == 0] #indices of benign images
        malig_idx =  [i for i in self.indices if self.y_data[i] == 1] #indices of malignant images

        self.pos_pairs = []
        for idx_list in [benign_idx, malig_idx]:
            #For each classification train NUM_TRAIN_SAMPLES pairs
            for _ in range((NUM_TRAIN_SAMPLES)):
                #Get samples
                i, j = random.sample(idx_list, 2)
                #Append to List
                self.pos_pairs.append((i,j))
        self.pos_labels = [1] *len(self.pos_pairs)

        # Sample negative pairs
        num_negatives = int(NUM_TRAIN_SAMPLES / 2)
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
        #Gets image
        filename = self.filenames[idx]
        image_path = os.path.join(self.x_data, filename)

        #Converts to tensor
        img = Image.open(image_path).convert("RGB")
        return self.transform(img)


    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        img1, img2 = self._load_image(i), self._load_image(j)
        label = torch.tensor(self.pair_labels[idx], dtype=torch.float32)
        return img1, img2, label
    
class ClassifyDataset(torch.utils.data.Dataset):
    """Dataset to help load data to train classifier, also will be used for test data and validation data. Just contains image and target"""
    def __init__(self, x_data, y_data, indices):
        from pathlib import Path
        self.x_data = Path(x_data)
        self.y_data = y_data
        self.indices = indices
        self.transform = self.transform = transforms.Compose([
            transforms.Resize((DIMENSIONS, DIMENSIONS)),
            transforms.ToTensor(),
            ])
        self.filenames = [f.name for f in self.x_data.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg']]
        assert len(self.filenames) == len(y_data), f"Number of images ({len(self.filenames)}) does not match number of labels ({len(y_data)})"

    def __len__(self):
        return len(self.indices)
        
    def _load_image(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.x_data, filename)
        img = Image.open(image_path).convert("RGB")
        return self.transform(img)
    def __getitem__(self, idx):
        img = self._load_image(idx)
        label = self.y_data[idx]
        return img, label


         