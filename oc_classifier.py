import os
import argparse
import librosa
import torch
import torch.nn.functional as F
import torch.optim as optim
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
    def __init__(self, protocol_file, dataset_dir):
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
        self.file_list = []
        self.label_list = []
        self.dataset_dir = dataset_dir

        # file_list is now the second column of the protocol file
        # label list is now the fifth column of the protocol file
        # read the protocol file

        with open(self.protocol_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.split(" ")
                self.file_list.append(line[1])
                self.label_list.append(line[4])

        self._length = len(self.file_list)
   
    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        """return feature and label of each audio file in the protocol file
        """
        audio_file = self.file_list[idx]
        file_path = os.path.join(self.dataset_dir, audio_file + ".flac")
        feature, _ = librosa.load(file_path, sr=None)
        label = self.label_list[idx]

        # Convert the list of features and labels to tensors
        feature_tensors = torch.tensor(feature, dtype=torch.float32)
        label_tensors = torch.tensor(label, dtype=torch.int64)
        # print(f"feature_tensors.shape = {feature_tensors.shape}")
        # print(f"label_tensors.shape = {label_tensors.shape}")
        
        return feature_tensors, label_tensors

    def collate_fn(self, batch):
        """pad the time series 1D"""
        pass


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='One-class classifier')
    parser.add_argument('--train_dataset_dir', type=str, default="/datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_train/flac",
                        help='Path to the dataset directory')
    parser.add_argument('--test_dataset_dir', type=str, default="/datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_eval/flac",
                        help='Path to the test dataset directory')
    
    # initialize xlsr and lcnn models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ssl = SSLModel(device)
    lcnn = lcnn_net(asoftmax=False).to(device)


    # load pretrained weights
    ssl.load_state_dict(torch.load("/datac/longnv/occm/ssl_0.pt"))
    lcnn.load_state_dict(torch.load("/datac/longnv/occm/lcnn_0.pt"))    
    lccn = DataParallel(lcnn)
    ssl = DataParallel(ssl)
    print("Pretrained weights loaded")
    
    audio_file = "/datac/longnv/audio_samples/ADD2023_T2_T_00000000.wav"
    audio_data, _ = librosa.load(audio_file, sr=None)
    emb = ssl(torch.Tensor(audio_data).unsqueeze(0).to("cuda"))
    emb = emb.unsqueeze(1)
    out = lcnn(emb)
    predicted = torch.argmax(out, dim=1)
    print(out)
    print(predicted)
    