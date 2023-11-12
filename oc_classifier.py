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
                    self.file_list.append(line[1])
                    self.label_list.append(line[5]) # bonafide or spoof
            self._length = len(self.file_list)
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
        feature, _ = librosa.load(file_path, sr=None)
        label = [1 if self.label_list[idx] == "spoof" else 0]

        # Convert the list of features and labels to tensors
        feature_tensors = torch.tensor(feature, dtype=torch.float32)
        label_tensors = torch.tensor(label, dtype=torch.int64)
        # print(f"feature_tensors.shape = {feature_tensors.shape}")
        # print(f"label_tensors.shape = {label_tensors.shape}")
        
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
            emb = encoder(emb)
            total_embeddings.append(emb)
    
    # reference embedding is the mean of all embeddings
    reference_embedding = torch.mean(torch.stack(total_embeddings), dim=0)
    
    # threshold is the maximum Euclidean distance between the reference embedding and all embeddings
    for emb in total_embeddings:
        distance = F.pairwise_distance(reference_embedding, emb, p=2)
        total_distances.append(distance)
    threshold = torch.max(torch.stack(total_distances))
    
    return reference_embedding, threshold

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='One-class classifier')
    parser.add_argument('--protocol_file', type=str, default="/datab/Dataset/ASVspoof/LA/ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
                        help='Path to the protocol file')
    parser.add_argument('--dataset_dir', type=str, default="/datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_train/flac",
                        help='Path to the dataset directory')
    args = parser.parse_args()
    
    # initialize xlsr and lcnn models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssl = SSLModel(device)
    senet = se_resnet34().to(device)

    # load pretrained weights
    ssl.load_state_dict(torch.load("/datac/longnv/occm/ssl_0.pt"))
    senet.load_state_dict(torch.load("/datac/longnv/occm/senet34_0.pt"))
    senet = DataParallel(senet)
    ssl = DataParallel(ssl)
    print("Pretrained weights loaded")
    
    # create a reference embedding & find a threshold
    print("Creating a reference embedding...")
    train_dataset = ASVDataset(args.protocol_file, args.dataset_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    reference_embedding, threshold = create_reference_embedding(ssl, senet, train_dataloader, device)

    print(f"reference_embedding.shape = {reference_embedding.shape}")
    print(f"threshold = {threshold}")

    # audio_file = "/datac/longnv/audio_samples/ADD2023_T2_T_00000000.wav"
    # audio_data, _ = librosa.load(audio_file, sr=None)
    # emb = ssl(torch.Tensor(audio_data).unsqueeze(0).to("cuda"))
    # emb = emb.unsqueeze(1)
    # emb = senet(emb)
 
    