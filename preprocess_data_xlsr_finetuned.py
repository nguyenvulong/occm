import os
import torch
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.xlsr import SSLModel

# from utils import extract_lfcc, extract_cwt, extract_ssqcwt, extract_bfcc, extract_cqcc, extract_lpc, extract_mfcc, extract_mel

# This version is about synthetic speech detection
# the code is for wav2vec features with finetuned model

# to be used with dataloader_v2.py
# input is now a raw audio file


class PFDataset(Dataset):
    def __init__(self, protocol_file, dataset_dir):
        """
        Protocol files
            database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.train.trl.txt
            database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.dev.trl.txt
            database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.eval.trl.txt
        
        Example of each protocol file
            LA_0079 CON_T_0000029 - CON spoof
            LA_0079 CON_T_0000069 - CON spoof
            LA_0098 LA_T_9497115 - - bonafide
            LA_0098 LA_T_9557645 - - bonafide
            LA_0098 LA_T_9737995 - - bonafide

        Depending on the track is "train", "dev", or "eval", we will use different protocol files in __init__
        As a result, __getitem__ will return the feature and label of each audio file in the protocol file

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

    
    def _preprocess_data(self, label_file, audio_dir ):
        """label file is a .npy file from PartialSpoof dataset
        >>> labels['CON_T_0025373'] # <class 'numpy.ndarray'>
        array(['0', '0', '0', '0', '1', '1', '0', '1', '1', '1', '1', '1', '1',
       '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0'], dtype='<U21')
        audio_dir is where all the audio files are stored
        
        Variables:
            segment_labels: segment-level labels (list)
            locations: locations of the end of segments (last sample #number)
            timestamped_words: word-level labels (list)
        
        """

 
        # loop through the audio files
        count = 0
        for audio_file in os.listdir(audio_dir):
            audio_name = audio_file[:-4]
            if not audio_file.endswith(".flac"):
                continue
            # skip the file if already processed
            # here I used a trick to check if the file is already processed
            # by checking if the first segment of the file exists
            
            if os.path.exists(
            os.path.join(self.dataset_dir, audio_name + "_0.pt")
            ) or os.path.exists(
            os.path.join(self.dataset_dir, audio_name + "_1.pt")
            ):
                print(f"File {audio_file} already processed. Skipping...")
                continue

            count += 1
            print(f"Processing {audio_file}, count = {count}")
            audio_file = os.path.join(audio_dir, audio_file)
            
            audio_data, sr = librosa.load(audio_file, sr=None)
            audio_length = len(audio_data)
            utterance_features = self.ssl.extract_feat(torch.Tensor(audio_data).unsqueeze(0).to("cuda"))
            print("utterance_features.shape: ", utterance_features.shape)
         
            start = 0
            end = audio_length
            
            

            
            try:
                segment_labels = dict_labels[audio_name]
                print("segment_labels: ", segment_labels)

            except:
                print(f"Label not found for {audio_name}")
                continue
            
            if len(segment_labels) == 0:
                print(f"**UTTERANCE**")

            # check whether the audio segment file contains the location
            # check if the segment contains the location (fusion point)
            # if yes, then label 1 (contains abrupt change), otherwise label 0
            if any(start <= location < end for location in locations):
                label = 1
            else:
                label = 0
            
            dataset_dict = {
            'feature': torch.tensor(utterance_features, dtype=torch.float32),
            # 'feature': torch.tensor(utterance_features, dtype=torch.cfloat), 
            'label': torch.tensor(label, dtype=torch.int64)
            }

            # and save the feature and label of the segment
            # to the dataset directory
            # format: dataset_dir/audio_name.pt
            torch.save(dataset_dict, os.path.join(self.dataset_dir, audio_name + "_" + str(label) +  ".pt"))
            
        print("Total number of audio files processed: ", count)
        
    def __len__(self):
        return self._length

    def __getitem__(self, idx):

        file_path = os.path.join(self.dataset_dir, self.file_list[idx] + ".flac")
        feature, _ = librosa.load(file_path, sr=None)
        # convert label "spoof" = 1 and bonafine = 0
        label = self.label_list[idx]
        if label == "spoof":
            label = 1
        else:
            label = 0
        feature = torch.tensor(feature, dtype=torch.float32)
        # print(f"feature.shape = {feature.shape}")
        return feature, label

    def collate_fn(self, batch):
        """pad the time series 1D"""
        # print("collate_fn")
        max_width = max(features.shape[0] for features, _ in batch)
        padded_batch_features = []
        for features, _ in batch:
            pad_width = max_width - features.shape[0]
            padded_features = F.pad(features, (0, pad_width), mode='constant', value=0)
            padded_batch_features.append(padded_features)
            
        labels = torch.tensor([label for _, label in batch])
        
        padded_batch_features = torch.stack(padded_batch_features, dim=0)
        return padded_batch_features, labels
    
    # def collate_fn(self, batch):
    #     # Find the maximum dimension along the second axis
    #     max_dim = max(features.shape[1] for features, _ in batch)
        
    #     # Pad sequences to match the maximum dimension
    #     padded_batch = [
    #         F.pad(features, (0, max_dim - features.shape[1]), mode='constant', value=0)
    #         for features, _ in batch
    #     ]
        
    #     # Stack the padded sequences along a new dimension (representing batch size)
    #     padded_sequences = torch.stack(padded_batch, dim=0)
        
    #     # Collect the labels
    #     labels = torch.tensor([label for _, label in batch])
        
    #     return padded_sequences, labels



if __name__== "__main__":
    d = "./v5_w2v2_train"
    print(f"Dataset: {d}, Extract Function: wav2vec2")
    # PS_LABEL_FILE = "./database/segment_labels/train_seglab_0.16.npy"
    # PS_AUDIO_DIR = "./database/train/con_wav"
    print(f"PS_LABEL_FILE: {PS_LABEL_FILE}, PS_AUDIO_DIR: {PS_AUDIO_DIR}")
    dataset = PFDataset(dataset_dir=d, extract_func="", preprocessed=False)

    pass