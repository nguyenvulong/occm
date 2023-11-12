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

from losses.custom_loss import triplet_loss ,compactness_loss, descriptiveness_loss, euclidean_distance_loss

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
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=test_dataset.collate_fn)

print("Instantiating model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ssl = SSLModel(device)
senet34 = se_resnet34().to(device)
# lcnn = lcnn_net(asoftmax=False).to(device)

optimizer = optim.Adam(list(ssl.parameters()) + list(senet34.parameters()), lr=0.0001)
# model = DataParallel(model)

ssl = DataParallel(ssl)
senet34 = DataParallel(senet34)
# lcnn = DataParallel(lcnn)

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
    ssl.train()
    senet34.train()
    # lcnn.train()
    
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device) # torch.Size([1, 3, 71648]), torch.Size([1, 3])
        # print(f"inputs.shape = {inputs.shape}, labels.shape = {labels.shape}")
        inputs = inputs.squeeze(0) # torch.Size([3, 81204])
        optimizer.zero_grad()

        # Forward pass
        outputs_ssl = ssl(inputs) # torch.Size([3, 191, 1024])
        outputs_ssl = outputs_ssl.unsqueeze(1) # torch.Size([3, 1, 191, 1024])
        outputs_senet34 = senet34(outputs_ssl) # torch.Size([3, 128])
        
        # Calculate the losses
        loss = triplet_loss(outputs_senet34)

        # print(f"loss = {float(loss)}")
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:
            print(f"[{epoch + 1}-{i + 1}] Train Loss: {running_loss / (i+1):.5f}")
            # write the loss to a file
            with open("loss.txt", "a") as f:
                f.write(f"epoch = {epoch + 1}-{i + 1}, Train Loss = {running_loss / (i+1):.5f} \n")
    # save the models after each epoch
    print("Saving the models...")
    torch.save(ssl.module.state_dict(), f"ssl_triplet_{epoch}.pt")
    torch.save(senet34.module.state_dict(), f"senet34_triplet_{epoch}.pt")