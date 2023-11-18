import os
import argparse
import librosa
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

import numpy as np
from torchattacks import PGD


from models.lcnn import *
from models.senet import *
from models.xlsr import *

from losses.custom_loss import compactness_loss, descriptiveness_loss, euclidean_distance_loss
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# to be used with one-class classifier
# input is now a raw audio file


class ASVDataset(Dataset):
    def __init__(self, protocol_file, dataset_dir, eval=False):
        """
        Protocol file for LA train
            /datab/Dataset/ASVspoof/LA/ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt
            Example
            LA_0079 LA_T_1138215 - - bonafide
        
        Protocol files for DF eval
            /datab/Dataset/ASVspoof/LA/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt
            the file only contains the file names
            
        metadata file for DF eval
            eval-package/keys/DF/CM/trial_metadata.txt
            Example
            LA_0043 DF_E_2000026 mp3m4a asvspoof A09 spoof notrim eval traditional_vocoder - - - -

        Args:
            dataset_dir (str): wav file directory
            extract_func (none): raw audio file
            preprocessed (True, optional): extract directly from raw audio files. Defaults to True.
        """
        self.protocol_file = protocol_file
        self.dataset_dir = dataset_dir
        self.file_list = []
        self.label_list = []  
        self.eval = eval
        # file_list is now the second column of the protocol file
        # label list is now the fifth column of the protocol file
        # read the protocol file

        if self.eval:
            # collect all files
            # note the difference between eval and train protocol file
            with open(self.protocol_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    line = line.split(" ")
                    self.file_list.append(line[0])
                    self.label_list.append("unknown") # bonafide or spoof
        else:
            # collect bona fide list only
            # for calculating `reference embedding`           
            with open(self.protocol_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    line = line.split(" ")
                    if line[4] == "bonafide":
                        self.file_list.append(line[1])
                        self.label_list.append(line[4]) # bonafide only
                        
        self._length = len(self.file_list)    
    
    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        """return feature and label of each audio file in the protocol file
        """
        audio_file = self.file_list[idx]
        file_path = os.path.join(self.dataset_dir, audio_file + ".flac")
        if not os.path.exists(file_path):
            file_path = os.path.join(self.dataset_dir, audio_file + ".wav")

        feature, _ = librosa.load(file_path, sr=None)
        feature_tensors = torch.tensor(feature, dtype=torch.float32)

        if self.eval == False:
            label = [1 if self.label_list[idx] == "spoof" else 0]
            # Convert the list of features and labels to tensors
            label_tensors = torch.tensor(label, dtype=torch.int64)
            return feature_tensors, label_tensors
        
        if self.eval:
            label = [1 if self.label_list[idx] == "spoof" else 0] # fake label
            label_tensors = torch.tensor(label, dtype=torch.int64)
            return feature_tensors, label_tensors
        

    def collate_fn(self, batch):
        """pad the time series 1D"""
        pass


def create_reference_embedding(extractor, encoder, dataloader, device):
    """Create reference embeddings for one-class classifier

    Args:
        extractor, encoder (nn.Module): pretrained models (e.g., XLSR, SE-ResNet34)
        dataloader (DataLoader): dataloader for the dataset

    Returns:
        torch.Tensor: reference embedding
    """
    # check whether the reference embedding and threshold already exist
    if os.path.exists("reference_embedding.pt") and os.path.exists("threshold.pt"):
        print("Loading reference embedding and threshold...")
        reference_embedding = torch.load("reference_embedding.pt")
        threshold = torch.load("threshold.pt")
        return reference_embedding, threshold
    
    print("Creating a reference embedding...")    
    extractor.eval()
    encoder.eval()
    total_embeddings = []
    total_distances = []
    
    with torch.no_grad():
        for _, (data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)
            emb = extractor(data)
            emb = emb.unsqueeze(1)
            emb = encoder(emb)[0]
            total_embeddings.append(emb)
    
    # reference embedding is the mean of all embeddings
    reference_embedding = torch.mean(torch.stack(total_embeddings), dim=0)
    
    # threshold is the maximum Euclidean distance between the reference embedding and all embeddings
    for emb in total_embeddings:
        distance = F.pairwise_distance(reference_embedding, emb, p=2)
        total_distances.append(distance)
    threshold = torch.max(torch.stack(total_distances))
    
    # save the reference embedding and threshold to a file
    torch.save(reference_embedding, "reference_embedding.pt")
    torch.save(threshold, "threshold.pt")
    return reference_embedding, threshold

def score_eval_set(extractor, encoder, dataloader, device, reference_embedding, threshold):
    """Score the evaluation set and save the scores to a file
       These scores will be used to calculate the EER.
    Args:
        extractor, encoder (nn.Module): pretrained models (e.g., XLSR, SE-ResNet34)
        dataloader (DataLoader): dataloader for the dataset
        reference_embedding (torch.Tensor): reference embedding
        threshold (float): threshold

    Returns:
        float: scores saved to a file
    """
    extractor.eval()
    encoder.eval()
    total_embeddings = []
    total_distances = []
    
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)
            emb = extractor(data)
            emb = emb.unsqueeze(1)
            emb = encoder(emb)[0]
            total_embeddings.append(emb)
            # total_labels.append(target)
            print(f"Processing file counts: {idx} ...")
    
    # calculate the distance between the reference embedding and all embeddings
    # write the scores to a file
    # each line contains a score and a label
    print("Scoring the evaluation set...")
    with open("scores.txt", "w") as f:
        for emb in total_embeddings:
            distance = F.pairwise_distance(reference_embedding, emb, p=2)
            total_distances.append(distance)
            if float(distance) > threshold:
                f.write(f"{float(distance)}, 1 \n")
            else:
                f.write(f"{float(distance)}, 0 \n")


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='One-class classifier')
    parser.add_argument('--pretrained-ssl', type=str, default="/datac/longnv/occm/ssl_triplet_1.pt",
                        help='Path to the pretrained weights')
    parser.add_argument('--pretrained-senet', type=str, default="/datac/longnv/occm/senet34_triplet_1.pt",
                        help='Path to the pretrained weights')
    parser.add_argument('--protocol_file', type=str, default="/datab/Dataset/ASVspoof/LA/ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
                        help='Path to the protocol file')
    parser.add_argument('--dataset_dir', type=str, default="/datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_train/flac",
                        help='Path to the dataset directory')
    parser.add_argument('--eval_protocol_file', type=str, default="/datab/Dataset/ASVspoof/LA/ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
                        help='Path to the protocol file')
    parser.add_argument('--eval_dataset_dir', type=str, default="/datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_eval/flac",
                        help='Path to the dataset directory')
    args = parser.parse_args()
    
    # initialize xlsr and lcnn models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssl = SSLModel(device)
    senet = se_resnet34().to(device)

    # load pretrained weights
    ssl.load_state_dict(torch.load(args.pretrained_ssl))
    senet.load_state_dict(torch.load(args.pretrained_senet))
    senet = DataParallel(senet)
    ssl = DataParallel(ssl)
    print("Pretrained weights loaded")
    
    # create a reference embedding & find a threshold
    train_dataset = ASVDataset(args.protocol_file, args.dataset_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    reference_embedding, threshold = create_reference_embedding(ssl, senet, train_dataloader, device)

    # score the evaluation set
    eval_dataset = ASVDataset(args.eval_protocol_file, args.eval_dataset_dir, eval=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)
    score_eval_set(ssl, senet, eval_dataloader, device, reference_embedding, threshold)
    
    print(f"reference_embedding.shape = {reference_embedding.shape}")
    print(f"threshold = {threshold}")
