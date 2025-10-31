import torch
from train import *
from model import *
from dataset import *
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from tqdm import tqdm

MODEL_PATH = "PatternAnalysis-2025/recognition/Siamese Network s4742616/model.pt"



def test_model(model, test_loader, device="cpu"):
    """Evaluate the model on the test dataset."""
    model.eval()  # set model to evaluation mode
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():  # disable gradient computation for faster inference
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model.predict(imgs)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


if __name__ == "__main__":
    # Load Model
    model = torch.load(MODEL_PATH, weights_only=False)
    print("Model Loaded Successfully")

    #  Get data
    val_data, test_data = get_test_val_data("cpu")
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    #Run Tests
    preds, labels = test_model(model, test_loader, device="cpu")


    #Output results
    #print(classification_report(preds, labels, target_names = ['benign', 'malignant']))
    print(f"Overall Accuracy: {accuracy_score(preds, labels)}")




