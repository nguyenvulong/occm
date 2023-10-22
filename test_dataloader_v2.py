import os
import argparse
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn import DataParallel
from torch.utils.data import DataLoader, random_split
from preprocess_data_xlsr import PFDataset
from sklearn.utils.class_weight import compute_class_weight

from models.lcnn import *
from models.cnn import *
from models.senet import *
from utils import *

# Train and Evaluate

# Arguments
print("Arguments...")
parser = argparse.ArgumentParser(description='Train a model on a dataset')
parser.add_argument('--dataset_dir', type=str, default="./lfcc_train",
                    help='Path to the dataset directory')
parser.add_argument('--test_dataset_dir', type=str, default="./lfcc_test",
                    help='Path to the test dataset directory')
parser.add_argument('--extract_func', type=str, default="none",
                    help='Name of the function to extract features from the dataset')
parser.add_argument('--model', type=str, default="lcnn_net")

# in case of finetuned, dataset_dir is the raw audio file directory instead of the extracted feature directory
parser.add_argument('--finetuned', action='store_true', default=False)
parser.add_argument('--train_protocol_file', type=str, default="./database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.train.trl.txt")
parser.add_argument('--test_protocol_file', type=str, default="./database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.dev.trl.txt")
args = parser.parse_args()

print("collate_fn...")
# collate function
def collate_fn(batch):
    max_width = max(features.shape[1] for features, _ in batch)
    max_height = max(features.shape[0] for features, _ in batch)
    padded_batch_features = []
    for features, _ in batch:
        pad_width = max_width - features.shape[1]
        pad_height = max_height - features.shape[0]
        padded_features = F.pad(features, (0, pad_width, 0, pad_height), mode='constant', value=0)
        padded_batch_features.append(padded_features)
    
    labels = torch.tensor([label for _, label in batch])
    
    padded_batch_features = torch.stack(padded_batch_features, dim=0)
    return padded_batch_features, labels

def collate_fn_total(batch):
    """pad the time series 1D"""
    max_width = max(features.shape[0] for features, _ in batch)
    padded_batch_features = []
    for features, _ in batch:
        pad_width = max_width - features.shape[0]
        padded_features = F.pad(features, (0, pad_width), mode='constant', value=0)
        padded_batch_features.append(padded_features)
        
    labels = torch.tensor([label for _, label in batch])
    
    padded_batch_features = torch.stack(padded_batch_features, dim=0)
    return padded_batch_features, labels


# Load the dataset
print("*************************************************")
print(f"Dataset dir = {args.dataset_dir}")
print(f"Test dataset dir = {args.test_dataset_dir}") 
print(f"extract_func = {args.extract_func}") 
print(f"model = {args.model}")
print(f"finetuned = {args.finetuned}") 
print(f"train_protocol_file = {args.train_protocol_file}") 
print(f"test_protocol_file = {args.test_protocol_file}")
print("*************************************************")

# Define the collate function
if args.finetuned:
    train_dataset = PFDataset(args.train_protocol_file, dataset_dir=args.dataset_dir)
    test_dataset = PFDataset(args.test_protocol_file, dataset_dir=args.test_dataset_dir)
    collate_func = collate_fn_total
else:
    train_dataset = PFDataset(dataset_dir=args.dataset_dir, extract_func=args.extract_func)
    test_dataset = PFDataset(dataset_dir=args.test_dataset_dir, extract_func=args.extract_func)
    collate_func = collate_fn

# Create dataloaders for training and validation
batch_size = 128

print("Creating dataloaders...")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, # for wav2vec2
                              collate_fn=collate_func)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                             collate_fn=collate_func)

print("Instantiating model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model == "lcnn_net":
    model = lcnn_net(asoftmax=False).to(device)
elif args.model == "lcnn_net_asoftmax":
    model = lcnn_net(asoftmax=True).to(device)
elif args.model == "cnn_net":
    model = cnn_net().to(device)
elif args.model == "cnn_net_with_attention":
    model = cnn_net_with_attention().to(device)
elif args.model == "se_resnet12":
    model = se_resnet12().to(device)
elif args.model == "se_resnet34":
    model = se_resnet34().to(device)
elif args.model == "total_cnn_net":
    model = total_cnn_net(device).to(device)
elif args.model == "total_resnet34":
    model = total_resnet34(device).to(device)


model = DataParallel(model)

# Class weights
print(f"Balanced class weights for train dataset: {args.dataset_dir}")
train_labels = []
for _, label in train_dataset:
    train_labels.append(label)
print("test_labels: ", np.unique(train_labels))
weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=np.array(train_labels))
print("weights: ", weights)
weights = torch.tensor(weights,dtype=torch.float).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=weights)
# criterion = nn.CrossEntropyLoss()

if args.model == "lcnn_net_asoftmax":
    criterion = AngleLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Instantiate the LR scheduler
# scheduler = StepLR(optimizer, step_size=1, gamma=0.5)  # Adjust the step_size and gamma as needed

# Number of epochs
num_epochs = 100

print("Starting training...")
# Training loop
best_val_acc = 0.0
best_test_acc = 0.0
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")

    # Training phase
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        if not args.finetuned:
            inputs = inputs.unsqueeze(1)  # Add channel dimension
        inputs = inputs.to(torch.float32)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate the loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Calculate training accuracy
        # outputs -> outputs[0] in case of AngleLoss
        if args.model == "lcnn_net_asoftmax":
            _, predicted = torch.max(outputs[0].data, 1)
        else:
            _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # Print statistics
        running_loss += loss.item()
        if i % 20 == 19:
            print(f"[{epoch + 1}, {i + 1}] Train Loss: {running_loss / (i+1):.3f}, \
                                           Train Acc: {(correct_train / total_train) * 100:.2f}")
    # Step LR scheduler
    # print("LR scheduled to {:.6f}".format(scheduler.get_lr()[0]))
    # scheduler.step()
    # Validation phase
    model.eval()  # Set the model to evaluation mode
    correct_val = 0
    total_val = 0
    val_loss = 0.0
    
    correct_test = 0
    total_test = 0
    test_loss = 0.0
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.to(torch.float32)
            if not args.finetuned:
                inputs = inputs.unsqueeze(1)  # Add channel dimension
            # Forward pass
            outputs = model(inputs)
            
            # Calculate the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # Calculate test accuracy
            # outputs -> outputs[0] in case of AngleLoss
            if args.model == "lcnn_net_asoftmax":
                _, predicted = torch.max(outputs[0].data, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
    # Calculate average training loss and accuracy for the epoch
    avg_train_loss = running_loss / len(train_dataloader)
    avg_train_acc = (correct_train / total_train) * 100

    print("***********************************************")
    print(f"Train Loss: {avg_train_loss:.3f}, Train Acc: {avg_train_acc:.2f}")
    print(f"Test Loss: {test_loss / len(test_dataloader):.3f}, Test Acc: {(correct_test / total_test) * 100:.2f}")
    test_acc = (correct_test / total_test) * 100
    # Save the best and the latest model only
    print("Saving the best model...")
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.module.state_dict(), args.model + "_" + os.path.basename(args.test_dataset_dir).split("_")[1] + "_best.pt")
    # print("Saving the latest model...")
    # torch.save(model.state_dict(), args.model + "_latest.pt")