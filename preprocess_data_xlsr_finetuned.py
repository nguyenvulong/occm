import os
import torch
import librosa
import random
import argparse

import numpy as np
import torch.nn.functional as F
from torchattacks import PGD
from torch.utils.data import Dataset
from data_utils_SSL import process_Rawboost_feature
# from audio_preprocess.audio_preprocess.denoise import DeNoise

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
        self._length = len(self.bonafide_indices)
        # self._denoiser = DeNoise()
        self._vocoded_dir = "/datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_vocoded"
        self.args = self._rawboost_args()
        
    def _rawboost_args(self):
        """Initialize params for args
        """
        parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
        parser.add_argument('--algo', type=int, default=3, 
                        help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                                5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .default=0]')

        # LnL_convolutive_noise parameters 
        parser.add_argument('--nBands', type=int, default=5, 
                        help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
        parser.add_argument('--minF', type=int, default=20, 
                        help='minimum centre frequency [Hz] of notch filter.[default=20] ')
        parser.add_argument('--maxF', type=int, default=8000, 
                        help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
        parser.add_argument('--minBW', type=int, default=100, 
                        help='minimum width [Hz] of filter.[default=100] ')
        parser.add_argument('--maxBW', type=int, default=1000, 
                        help='maximum width [Hz] of filter.[default=1000] ')
        parser.add_argument('--minCoeff', type=int, default=10, 
                        help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
        parser.add_argument('--maxCoeff', type=int, default=100, 
                        help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
        parser.add_argument('--minG', type=int, default=0, 
                        help='minimum gain factor of linear component.[default=0]')
        parser.add_argument('--maxG', type=int, default=0, 
                        help='maximum gain factor of linear component.[default=0]')
        parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                        help=' minimum gain difference between linear and non-linear components.[default=5]')
        parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                        help=' maximum gain difference between linear and non-linear components.[default=20]')
        parser.add_argument('--N_f', type=int, default=5, 
                        help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

        # ISD_additive_noise parameters
        parser.add_argument('--P', type=int, default=10, 
                        help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
        parser.add_argument('--g_sd', type=int, default=2, 
                        help='gain parameters > 0. [default=2]')

        # SSI_additive_noise parameters
        parser.add_argument('--SNRmin', type=int, default=10, 
                        help='Minimum SNR value for coloured additive noise.[defaul=10]')
        parser.add_argument('--SNRmax', type=int, default=40, 
                        help='Maximum SNR value for coloured additive noise.[defaul=40]')
        args = parser.parse_args()
        return args
        
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
    
    def _get_vocoded_files(self, bonafide):
        """Get vocoded file names, these files are generated by augmenting the 
        bonafide file.
        Examples of vocoded filepaths:
            /datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_vocoded/hifigan_LA_T_1138215.wav
            /datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_vocoded/hn-sinc-nsf-hifi_LA_T_1138215.wav
            /datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_vocoded/hn-sinc-nsf_LA_T_1138215.wav
            /datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_vocoded/melgan_LA_T_1138215.wav
            /datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_vocoded/waveglow_LA_T_1138215.wav
        The vocoded file names have the prefix of vocoder names and the bonafide file names
        For example, LA_T_1138215 is the bonafide file name,
        then the vocoded file names are:
            hifigan_LA_T_1138215
            hn-sinc-nsf-hifi_LA_T_1138215
            hn-sinc-nsf_LA_T_1138215
            melgan_LA_T_1138215
            waveglow_LA_T_1138215

        Args:
            bonafide (str): name of the bonafide file
        """
        # return a list of vocoded file names
        vocoder_names = ["hifigan", "hn-sinc-nsf-hifi", "hn-sinc-nsf", "melgan", "waveglow"]
        return [f"{vocoder_name}_{bonafide}" for vocoder_name in vocoder_names]
    
    def _get_files(self, idx):
        """Get files for training.
        If the idx label is bonafide, then select 5 other files from the bonafide list
        and 1 file from the spoof list.
        Because self._length is the number of bonafide files.
        idx always points to a bonafide file.
        Args:
            idx (int): index of the file in the protocol file
        """
        bona_files = self._get_random_files(self.bonafide_indices, idx, 5)
        spoof_files = self._get_random_files(self.spoof_indices, None, 1)
        return {
            'bona1': self.file_list[idx],  # The indexed file is bona1
            'bona2': bona_files[0],        # The additional bonafide file is bona2
            'bona3': bona_files[1],        # The additional bonafide file is bona3
            'bona4': bona_files[2],        # The additional bonafide file is bona4
            'bona5': bona_files[3],        # The additional bonafide file is bona5
            'bona6': bona_files[4],        # The additional bonafide file is bona6
            'spoof1': spoof_files[0],      # The first spoof file
        }
    
    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        """return feature and label of each audio file in the protocol file
        """
        # Because self._length is the number of bonafide files
        # This idx always points to a bonafide file
        # Get a list of files to be used
        file_assignments = self._get_files(idx)
        # print(f"file_assignments = {sorted(file_assignments)}")
        features = []
        labels = []
        max_length = 0
        
        # First get the features and labels from the 6 bonafide files
        # And 1 spoof file
        for key, audio_file in sorted(file_assignments.items()):
            file_path = os.path.join(self.dataset_dir, audio_file + ".flac")
            feature, sr = librosa.load(file_path, sr=None)
            # rawboost augmentation, algo=4 is the series of 1, 2, 3
            feature = process_Rawboost_feature(feature, sr, self.args, 4)
            max_length = max(max_length, feature.shape[0])
                       
            # Convert label "spoof" = 1 and "bonafide" = 0
            label = 1 if key.startswith("spoof") else 0
           
            features.append(feature)
            labels.append(label)
            
        # Get the vocoded files for 5 additional spoof files
        # And append them to the list of features and labels
        # bona1 is the original file
        vocoded_files = self._get_vocoded_files(file_assignments['bona1'])
        for vocoded_file in vocoded_files:
            file_path = os.path.join(self._vocoded_dir, vocoded_file + ".wav")
            feature, sr = librosa.load(file_path, sr=None)
            # rawboost augmentation, algo=4 is the series of 1, 2, 3
            feature = process_Rawboost_feature(feature, sr, self.args, 4) 
            max_length = max(max_length, feature.shape[0])
            label = 1
            features.append(feature)
            labels.append(label)
        
        # Pad the features to have the same length
        features_padded = []
        for feature in features:
            # You might want to specify the type of padding, e.g., zero padding
            feature_padded = np.pad(feature, (0, max_length - len(feature)), mode='constant')
            features_padded.append(feature_padded)
        
        # Convert the list of features and labels to tensors
        features = np.array(features_padded)
        labels = np.array(labels)
        feature_tensors = torch.tensor(features, dtype=torch.float32)
        label_tensors = torch.tensor(labels, dtype=torch.int64)
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