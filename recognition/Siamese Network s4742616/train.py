from dataset import *
from model import *
import torch
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt


RANDOM_STATE = 354


PRE_PROCCESSED = True #Flag to see if we've already pre-processed data 
MODEL_PATH = "PatternAnalysis-2025/recognition/Siamese Network s4742616/model.pt"
def train_model(model, num_epochs = 10, lr = 1e-4):
    embed_loss = train_model_embeddings(model, num_epochs = 0, lr = lr)
    classify_loss = train_model_classifier(model, num_epochs = num_epochs, lr = lr)
    return embed_loss, classify_loss

def train_model_embeddings(model, num_epochs = 10, lr = 1e-4):
    #loss = SOMELOSSFUNCTION
    optimizer = optim.Adam(model.parameters(), lr = lr)
    lossfunc = ContrastiveLoss()
    losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for img1, img2, labels in tqdm(train_loader, desc = f"(Embeddings) Epoch {epoch+1}/{num_epochs}"):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            optimizer.zero_grad()

            output1, output2 = model(img1, img2)
            loss = lossfunc(output1, output2, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * img1.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
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
    optimizer = optim.Adam(model.parameters(), lr = lr)
    lossfunc = nn.BCELoss()
    losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for img1, img2, labels in tqdm(train_loader, desc = f"(Classifier) Epoch {epoch+1}/{num_epochs}"):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            labels = labels.float().unsqueeze(-1) #makes same shape as output

            optimizer.zero_grad()

            output1 = model(img1, classify = True)
            loss = lossfunc(output1, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * img1.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
        losses.append(epoch_loss)
    return losses

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Loads and preproccesses datasets
    print(f"Training on Device: {device}")
    #Loads Data
    train = get_data(device) 
    #Sets up dataset loaders
    train_loader = DataLoader(train, batch_size = 16, shuffle = True)
    print("Data Loaded Successfully")

    #load models
    model = SiameseNetwork().to(device)
    

    embed_loss, classify_loss = train_model(model)

    plt.plot(embed_loss, main = "Loss for Feature Extraction", ylab = "Loss", xlab = "Epoch")

    plt.savefig("Embedding_Loss.png")
    plt.plot(classify_loss, main = "Loss for Classification", ylab = "Loss", xlab = "Epoch")
    torch.save(model, MODEL_PATH)

