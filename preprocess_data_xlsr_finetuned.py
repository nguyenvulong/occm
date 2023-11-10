import os
import torch
import librosa
import random

import numpy as np
import torch.nn.functional as F
from torchattacks import PGD
from torch.utils.data import Dataset
from audio_preprocess.audio_preprocess.denoise import DeNoise

# from utils import extract_lfcc, extract_cwt, extract_ssqcwt, extract_bfcc, extract_cqcc, extract_lpc, extract_mfcc, extract_mel

# This version is about synthetic speech detection
# the code is for wav2vec features with finetuned model

# to be used with dataloader_v2.py
# input is now a raw audio file


class PFDataset(Dataset):
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
                
        # Caching the indices of each label for quick access
        self.spoof_indices = [i for i, label in enumerate(self.label_list) if label == 'spoof']
        self.bonafide_indices = [i for i, label in enumerate(self.label_list) if label == 'bonafide']
        self._length = len(self.file_list)
        self._denoiser = DeNoise()
    
    def _denoise(self, audio_data):
        """denoise the audio data
        """
        denoiser = DeNoise()
        return denoiser.process(audio_data)
        
    def _adversarial_attack(self, audio_data, model):
        """adversarial attack
        """
        atk = PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
        return atk(audio_data, torch.tensor([1])) # 1 is the target class of spoof
    
    def _get_random_files(self, indices_list, exclude_idx, number_needed):
        """Get random files from the list of indices, excluding the exclude_idx

        Args:
            indices_list (int): index list
            exclude_idx (int): index to exclude
            number_needed (int): number of files needed

        Raises:
            ValueError: _description_

        Returns:
            _type_: dictionary of files
        """
        if exclude_idx is not None:
            possible_indices = list(set(indices_list) - {exclude_idx})
        else:
            possible_indices = list(indices_list)
        if len(possible_indices) < number_needed:
            raise ValueError("Not enough files to select from.")
        selected_indices = random.sample(possible_indices, k=number_needed)
        return [self.file_list[idx] for idx in selected_indices]
    
    def _get_files(self, idx):
        """Check provided index for 'bonafide' or 'spoof'"""
        label = self.label_list[idx]
        if label == 'bonafide':
            bona_files = self._get_random_files(self.bonafide_indices, idx, 1)
            spoof_files = self._get_random_files(self.spoof_indices, None, 4)
            return {
                'bona1': self.file_list[idx],  # The indexed file is bona1
                'bona2': bona_files[0],        # The additional bonafide file is bona2
                'spoof1': spoof_files[0],      # The first spoof file
                'spoof2': spoof_files[1],      # The second spoof file
                'spoof3': spoof_files[2],      # The third spoof file
                'spoof4': spoof_files[3]       # The fourth spoof file
            }
        elif label == 'spoof':
            bona_files = self._get_random_files(self.bonafide_indices, None, 2)
            spoof_files = self._get_random_files(self.spoof_indices, idx, 3)
            return {
                'spoof1': self.file_list[idx],  # The indexed file is spoof1
                'spoof2': spoof_files[0],      # The first additional spoof file
                'spoof3': spoof_files[1],      # The second additional spoof file
                'spoof4': spoof_files[2],      # The third additional spoof file
                'bona1': bona_files[0],        # The first bonafide file
                'bona2': bona_files[1]         # The second bonafide file
            }
        else:
            raise ValueError(f"Invalid label at index {idx}")


    
    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        """return feature and label of each audio file in the protocol file
        """
        # TODO: how to distribute the samples in each batch
        
        # Get a list of files to be used
        file_assignments = self._get_files(idx)
        
        # Finalize the list of files to be used
        # bona3 and bona4 are the same as bona1 and bona2
        # but later on, we will denoise bona1 and bona2
        # to create bona3 and bona4
        file_assignments['bona3'] = file_assignments['bona1']
        file_assignments['bona4'] = file_assignments['bona2']
        
        # print(f"file_assignments = {sorted(file_assignments)}")
        # Get features and labels from the files
        # And stack them into tensors
        features = []
        labels = []
        max_length = 0
        # ['bona1', 'bona2', 'bona3', 'bona4', 'spoof1', 'spoof2', 'spoof3', 'spoof4']
        for key, audio_file in sorted(file_assignments.items()):
            # print(f"label = {key}, audio_file = {audio_file}")
            file_path = os.path.join(self.dataset_dir, audio_file + ".flac")
            feature, _ = librosa.load(file_path, sr=None)
            
            if key == "bona3" or key == "bona4":
                feature = self._denoise(feature)
            max_length = max(max_length, feature.shape[0])
                       
            # Convert label "spoof" = 1 and "bonafide" = 0
            label = 1 if key.startswith("spoof") else 0
           
            features.append(feature)
            labels.append(label)
                 
        # Pad the features to have the same length
        features_padded = []
        for feature in features:
            # You might want to specify the type of padding, e.g., zero padding
            feature_padded = np.pad(feature, (0, max_length - len(feature)), mode='constant')
            features_padded.append(feature_padded)
            
        features = np.array(features_padded)
        labels = np.array(labels)
        # Convert the list of features and labels to tensors
        feature_tensors = torch.tensor(features, dtype=torch.float32)
        label_tensors = torch.tensor(labels, dtype=torch.int64)
        # print(f"feature_tensors.shape = {feature_tensors.shape}")
        # print(f"label_tensors.shape = {label_tensors.shape}")
        
        return feature_tensors, label_tensors

    def collate_fn(self, batch):
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


if __name__== "__main__":
    pass