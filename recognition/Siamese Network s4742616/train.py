from dataset import *
from model import *
import torch
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

#Some parameters, can be changed if needed
RANDOM_STATE = 354
NUM_WORKERS = 2
BATCH_SIZE = 32
MODEL_PATH = "PatternAnalysis-2025/recognition/Siamese Network s4742616/model.pt"

def train_model(model, num_epochs = 10, lr = 1e-4):
    """Function to train the model"""
    #Get feature extraction training done (Computing similarity)
    embed_loss = train_model_embeddings(model, num_epochs = 10, lr = lr)
    #Then train the classifier
    classify_loss = train_model_classifier(model, num_epochs = num_epochs, lr = lr)
    return embed_loss, classify_loss

def train_model_embeddings(model, num_epochs = 10, lr = 1e-4):

    #Initial SetUp
    optimizer = optim.Adam(model.parameters(), lr = lr)
    lossfunc = ContrastiveLoss()
    losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for img1, img2, labels in tqdm(train_loader_embed, desc = f"(Embeddings) Epoch {epoch+1}/{num_epochs}"):
            #Load Images
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            optimizer.zero_grad()

            #Get outputs
            output1, output2 = model(img1, img2)
            #Compute loss and step in direction of negative gradient
            loss = lossfunc(output1, output2, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * img1.size(0)
        #Some metrics for visualisation
        epoch_loss = running_loss / len(train_loader_embed.dataset)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
        losses.append(epoch_loss)
    return losses

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Compute Euclidean distance between embeddings
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        
        # Contrastive loss formula
        loss = (1 - label) * 0.5 * torch.pow(euclidean_distance, 2) + \
               label * 0.5 * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss.mean()
    
def train_model_classifier(model, num_epochs = 10, lr = 1e-4):
    """Optimization of classifier"""
    optimizer = optim.Adam(model.classifier.parameters(), lr = lr)
    weights = torch.tensor([0.95/0.05])
    lossfunc = nn.BCEWithLogitsLoss(pos_weight = weights)
    losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for img, label in tqdm(train_loader_class, desc = f"(Classifier) Epoch {epoch+1}/{num_epochs}"):
            img, label = img.to(device), label.to(device)

            label = label.float()

            optimizer.zero_grad()

            output1 = model(img, classify = True)
            loss = lossfunc(output1, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * img.size(0)
        epoch_loss = running_loss / len(train_loader_class.dataset)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
        losses.append(epoch_loss)
    return losses

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Loads and preproccesses datasets
    print(f"Training on Device: {device}")
    #Loads Data
    train_embed, train_class = get_data(device) 
    #Sets up dataset loaders
    train_loader_embed = DataLoader(train_embed, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
    train_loader_class = DataLoader(train_class, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
    print("Data Loaded Successfully")

    #load models
    model = SiameseNetwork().to(device)
    

    embed_loss, classify_loss = train_model(model)

    plt.plot(embed_loss)
    plt.title("Loss vs Epoch for Feature Extraction")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig("Embedding_Loss.png")
    plt.clf()  # Clears the previous plot

    plt.plot(classify_loss)
    plt.title("Loss vs Epoch for Classification")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("Classification_Loss.png")
    torch.save(model, MODEL_PATH)

