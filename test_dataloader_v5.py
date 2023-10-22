
import os
import argparse
import pandas as pd
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn import DataParallel
from torch.utils.data import DataLoader, random_split


from evaluate_metrics import compute_eer
# from preprocess_data_dsp import PFDataset
from preprocess_data_xlsr import PFDataset
# from preprocess_data_xlsr_finetuned import PFDataset
from models.lcnn import *
from models.cnn import *
from models.senet import *
from utils import *

# Evaluate only, calculate EER
# Additional dimension is due to the label, see `preprocess_data_xlsr.py`

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

# in case of eval
parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('--pretrained_model', type=str, default="model.pt")

# in case of finetuned, dataset_dir is the raw audio file directory instead of the extracted feature directory
parser.add_argument('--finetuned', action='store_true', default=False)
parser.add_argument('--train_protocol_file', type=str, default="./database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.train.trl.txt")
parser.add_argument('--test_protocol_file', type=str, default="./database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.dev.trl.txt")
args = parser.parse_args()


print("collate_fn...")
# collate function
def collate_fn(batch):
    max_width = max(features.shape[1] for features, _, _ in batch)
    max_height = max(features.shape[0] for features, _, _ in batch)
    padded_batch_features = []
    for features, _, _ in batch:
        pad_width = max_width - features.shape[1]
        pad_height = max_height - features.shape[0]
        padded_features = F.pad(features, (0, pad_width, 0, pad_height), mode='constant', value=0)
        padded_batch_features.append(padded_features)
    
    labels = torch.tensor([label for _, label, _ in batch])
    fnames = [fname for _, _, fname in batch]
    padded_batch_features = torch.stack(padded_batch_features, dim=0)
    return padded_batch_features, labels, fnames

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
    # train_dataset = PFDataset(dataset_dir=args.dataset_dir, extract_func=args.extract_func)
    test_dataset = PFDataset(dataset_dir=args.test_dataset_dir, extract_func=args.extract_func, eval=args.eval)
    collate_func = collate_fn

# Create dataloaders for training and validation
batch_size = 128

print("Creating dataloaders...")
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, # for wav2vec2
#                               collate_fn=collate_func)
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

# Set the model to evaluation mode
model.eval()  
# load model from the best model
print("Loading model weights...")
model.load_state_dict(torch.load(args.pretrained_model), strict=True)
model = DataParallel(model)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
if args.model == "lcnn_net_asoftmax":
    criterion = AngleLoss()
# Validation phase

correct_test = 0
total_test = 0
test_loss = 0.0
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN values found in parameter: {name}")
print("Evaluating on test set...")

score_file = args.model + "_" + os.path.basename(args.test_dataset_dir).split("_")[1] + "_eval_scores.txt"
with torch.no_grad():
    for data in test_dataloader:
        inputs, labels, fnames = data[0].to(device), data[1].to(device), data[2]
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
        
        with open(score_file, "a") as f:
            for i in range(len(fnames)):
                if args.model == "lcnn_net_asoftmax":
                    f.write(f"{fnames[i]} {outputs[0][i][0]}\n")
                else:
                    f.write(f"{fnames[i]} {outputs[i][0]}\n")
        

print("***********************************************")
print(f"Test Loss: {test_loss / len(test_dataloader):.3f}, Test Acc: {(correct_test / total_test) * 100:.2f}")
print("***********************************************")



def calculate_EER(score_file=score_file):
    """
    Step:
        - load protocol file
        - load score file
        - calculate EER
    """
    pro_columns = ["sid", "utt","phy", "attack", "label"]
    eval_protocol_file = pd.read_csv(args.test_protocol_file, sep=" ", header=None)
    eval_protocol_file.columns = pro_columns
    
    score_file = pd.read_csv(score_file, sep=" ", header=None)
    score_file.columns = ["utt", "score"]
    
    res = pd.merge(eval_protocol_file, score_file, on="utt")
    spoof_scores = res[res["label"] == "spoof"]["score"].values
    bonafide_scores = res[res["label"] == "bonafide"]["score"].values
    
    eer, threshold = compute_eer(bonafide_scores, spoof_scores)
    print(f"EER = {eer*100.0}, threshold = {threshold}")
    
calculate_EER(score_file=score_file)