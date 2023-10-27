
import torch.nn as nn
import fairseq
import os
import sys
from senet import *
from lcnn import *


BASE_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("losses")
from custom_loss import compactness_loss, descriptiveness_loss




class SSLModel(nn.Module):
    def __init__(self,device):
        super(SSLModel, self).__init__()
        
        cp_path = os.path.join(BASE_DIR,'/datac/longnv/SSL_Anti-spoofing/pretrained/xlsr2_300m.pt')
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.model.to(device)
        self.model.eval()
        self.out_dim = 1024


    def extract_feat(self, input_data):        

        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
        # [batch, length, dim]
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        # print(emb.shape)
        return emb



class OCCM(nn.Module):
    def __init__(self, device):
        super(OCCM, self).__init__()
        self.frontend = SSLModel(device)
        self.senet34_branch = se_resnet34().to(device)
        self.lcnn_branch = lcnn_net(asoftmax=False).to(device)
        
    def forward(self, x):
        x = self.frontend.extract_feat(x)
        x = x.unsqueeze(1)
        senet34_output = self.senet34_branch(x)
        lcnn_output = self.lcnn_branch(x)
        return senet34_output, lcnn_output

if __name__ == "__main__":
    model = OCCM("cuda")
    audio_file = "/datac/longnv/audio_samples/ADD2023_T2_T_00000000.wav"
    import librosa
    import torch
    audio_data, _ = librosa.load(audio_file, sr=None)
    x,y = model(torch.Tensor(audio_data).unsqueeze(0).to("cuda"))
    print(x.shape)
    print(y)
