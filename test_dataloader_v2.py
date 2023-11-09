import os
import argparse
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn import DataParallel
from torch.utils.data import DataLoader, random_split
from preprocess_data_xlsr_finetuned import PFDataset
from sklearn.utils.class_weight import compute_class_weight

from models.lcnn import *
from models.senet import *
from models.xlsr import *

from losses.custom_loss import compactness_loss, descriptiveness_loss

# Train and Evaluate

# Arguments
print("Arguments...")
parser = argparse.ArgumentParser(description='Train a model on a dataset')
parser.add_argument('--train_dataset_dir', type=str, default="/datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_train/flac",
                    help='Path to the dataset directory')
parser.add_argument('--test_dataset_dir', type=str, default="/datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_eval/flac",
                    help='Path to the test dataset directory')
parser.add_argument('--model', type=str, default="ssl_resnet34")

# in case of finetuned, dataset_dir is the raw audio file directory instead of the extracted feature directory
parser.add_argument('--finetuned', action='store_true', default=False)
parser.add_argument('--train_protocol_file', type=str, default="/datab/Dataset/ASVspoof/LA/ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
parser.add_argument('--test_protocol_file', type=str, default="/datab/Dataset/ASVspoof/LA/ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")
args = parser.parse_args()


# Load the dataset
print("*************************************************")
print(f"Train dataset dir = {args.train_dataset_dir}")
print(f"Test dataset dir = {args.test_dataset_dir}") 
print(f"model = {args.model}")
print(f"finetuned = {args.finetuned}") 
print(f"train_protocol_file = {args.train_protocol_file}") 
print(f"test_protocol_file = {args.test_protocol_file}")
print("*************************************************")

# Define the collate function

train_dataset = PFDataset(args.train_protocol_file, dataset_dir=args.train_dataset_dir)
test_dataset = PFDataset(args.test_protocol_file, dataset_dir=args.test_dataset_dir)

# Create dataloaders for training and validation
batch_size = 1

print("Creating dataloaders...")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=train_dataset.collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=test_dataset.collate_fn)

print("Instantiating model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ssl = SSLModel(device)
senet34 = se_resnet34()
lcnn = lcnn_net()

optimizer = optim.Adam(list(ssl.parameters()) + list(senet34.parameters()) + list(lcnn.parameters()), lr=0.0001)
# model = DataParallel(model)

# Define the loss functions
# Descriptiveness loss is CrossEntropyLoss
# Compactness loss is mahalanobis distance

# criterion_descriptiveness = descriptiveness_loss()
# criterion_compactness = compactness_loss()


# Also consider criterion = nn.BCEWithLogitsLoss()
# Note that nn.CrosEntropyLoss() expects raw logits from the last layer 
# and target labels are class indices
# while nn.BCEWithLogitsLoss() combines a Sigmoid layer and the BCELoss in one single class,
# so it expects raw logits from the last layer,    
# and target labels are class probabilities

# if args.model == "lcnn_net_asoftmax":
#     criterion = AngleLoss()



# Number of epochs
num_epochs = 100

print("Training starts...")
# Training loop
best_val_acc = 0.0
best_test_acc = 0.0
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")

    # Training phase
    ssl.eval()
    senet34.train()
    lcnn.train()
    
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        print("labels: ", labels)
        # print(f"inputs.shape = {inputs.shape}")

        if not args.finetuned:
            inputs = inputs.unsqueeze(1)  # Add channel dimension
        inputs = inputs.to(torch.float32)
        optimizer.zero_grad()

        # Forward pass
        outputs_ssl = ssl(inputs)
        outputs_senet34 = senet34(outputs_ssl)
        outputs_lcnn = lcnn(outputs_ssl)
        
        # Calculate the loss
        loss = compactness_loss(outputs_senet34) + descriptiveness_loss(outputs_lcnn, labels)
        
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
        torch.save(model.module.state_dict(), args.model  + "_best.pt")