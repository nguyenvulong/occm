import random
from typing import Union

import numpy as np
# import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
import fairseq
import os
from torch.nn import DataParallel

# import model.resnet as resnet
# from model.loss_metrics import supcon_loss

___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

############################
## FOR fine-tuned SSL MODEL
############################

BASE_DIR=os.path.dirname(os.path.abspath(__file__))

class SSLModel(nn.Module):
    def __init__(self, device):
        super(SSLModel, self).__init__()
        
        cp_path = os.path.join(BASE_DIR,'/datac/longnv/SSL_Anti-spoofing/pretrained/xlsr2_300m.pt')
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.model.to(device)
        self.model.eval()
        self.out_dim = 1024
        
        return

    def extract_feat(self, input_data):        

        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
        # [batch, length, dim]
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        # print(emb.shape)
        return emb
    
    
# Test the model in main function

if __name__ == "__main__":
    m = SSLModel("cuda")
    audio_file = "/datac/longnv/audio_samples/ADD2023_T2_T_00000000.wav"
    import librosa
    import torch
    audio_data, sr = librosa.load(audio_file, sr=None)
    emb = m.extract_feat(torch.Tensor(audio_data).unsqueeze(0).to("cuda"))
    print(emb.shape)