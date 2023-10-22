import os
import torch
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.ssl import SSLModel

from utils import extract_lfcc, extract_cwt, extract_ssqcwt, extract_bfcc, extract_cqcc, extract_lpc, extract_mfcc, extract_mel

# This version 5 of preprocess_data.py is about synthetic speech detection
# we will use the Partial Spoof dataset 
# All audio files are used for training and testing
# the code is for wav2vec features

# to be used with dataloader_v2.py

PS_LABEL_FILE = "./database/segment_labels/train_seglab_0.16.npy"
PS_AUDIO_DIR = "./database/train/con_wav"

class PFDataset(Dataset):
    def __init__(self, dataset_dir, extract_func, eval=False, preprocessed=True):
        self.label_file = PS_LABEL_FILE
        self.audio_dir = PS_AUDIO_DIR
        self.dataset_dir = dataset_dir
        self.extract_func = extract_func
        self.eval = eval
        self.preprocessed = preprocessed
        self.ssl = SSLModel(device="cuda")
        if self.preprocessed:
            self.file_list = [filename for filename in os.listdir(self.dataset_dir) if filename.endswith(".pt")]
            print(f"File List sorting in {self.dataset_dir}...")
            self.file_list = sorted(self.file_list)
            self._length = len(self.file_list)
            print("File List sorted.")
            pass
        else:
            print(f"Preprocessing in {self.audio_dir}...")
            self._preprocess_data(self.label_file, self.audio_dir)
    def _convert_labels(self, segment_labels, total_samples):
        """Converts the segment_labels to samples (location) based on total_samples
        >>> labels['CON_T_0025373']
        array(['0', '0', '0', '0', '1', '1', '0', '1', '1', '1', '1', '1', '1'], dtype='<U21')
        Then we will have 13 segment-level labels for this audio file.
        Also we have 4 larger segments for this audio file. They are
        segment 1 = ['0', '0', '0', '0']
        segment 2 = ['1', '1']
        segment 3 = ['0']
        segment 4 = ['1', '1', '1', '1', '1', '1']
        
        We can get the 4 locations by this formula
        ((index of the last element of a segment + 1) * total_samples) / len(labels)
        
        For example, segment 1 is ['0', '0', '0', '0'] and the last element index is 3.
        Assume that the total_samples are 30000, then the location of the last element is
        ((3 + 1) * 30000) / 13 = 9230 (integer only)
        
        In this work we only care about the locations except the last one, since such locations
        represent the end of a segment (i.e., change from '0' to '1' or vice versa). The last location
        does not represent the end of a segment, so we will not use it.
        
        """
        locations = []
        for i, label in enumerate(segment_labels):
            if i == len(segment_labels) - 1:
                break
            if label != segment_labels[i+1]:
                locations.append(((i+1) * total_samples) // len(segment_labels))
        return locations


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

        # get the labels dictionary
        dict_labels = np.load(label_file, allow_pickle=True).item()

        # load the whisper model
        print("Loading the whisper model...")
        # whisper_model = whisper.load_model(WHISPER_MODEL_FILE, device="cuda")

        # loop through the audio files
        count = 0
        for audio_file in os.listdir(audio_dir):
            audio_name = audio_file[:-4]
            if not audio_file.endswith(".wav"):
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
            locations = self._convert_labels(segment_labels, audio_length)

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
        if self.preprocessed:
            file_path = os.path.join(self.dataset_dir, self.file_list[idx])
            file_name = os.path.basename(file_path)
            dataset_dict = torch.load(file_path)
            feature = dataset_dict['feature']
            feature = feature.squeeze(0).to("cuda")
            label = dataset_dict['label']
            if self.eval:
                return feature, label, "_".join(file_name.split("_")[:-1])
            else:
                return feature, label

        
    def collate_fn(self, batch):
        # Find the maximum dimension along the second axis
        max_dim = max(features.shape[1] for features, _ in batch)
        
        # Pad sequences to match the maximum dimension
        padded_batch = [
            F.pad(features, (0, max_dim - features.shape[1]), mode='constant', value=0)
            for features, _ in batch
        ]
        
        # Stack the padded sequences along a new dimension (representing batch size)
        padded_sequences = torch.stack(padded_batch, dim=0)
        
        # Collect the labels
        labels = torch.tensor([label for _, label in batch])
        
        return padded_sequences, labels



if __name__== "__main__":
    d = "./v5_w2v2_train"
    print(f"Dataset: {d}, Extract Function: wav2vec2")
    # PS_LABEL_FILE = "./database/segment_labels/train_seglab_0.16.npy"
    # PS_AUDIO_DIR = "./database/train/con_wav"
    print(f"PS_LABEL_FILE: {PS_LABEL_FILE}, PS_AUDIO_DIR: {PS_AUDIO_DIR}")
    dataset = PFDataset(dataset_dir=d, extract_func="", preprocessed=False)

    pass