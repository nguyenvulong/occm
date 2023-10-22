import numpy as np
import torch
import torchvision.transforms as transforms

from ssqueezepy import ssq_cwt

from scipy.io.wavfile import read
from spafe.features.lfcc import lfcc
from spafe.features.lpc import lpc
from spafe.features.bfcc import bfcc
from spafe.features.cqcc import cqcc
from spafe.features.lpc import lpcc
from spafe.features.mfcc import mfcc
from spafe.features.mfcc import mel_spectrogram

from spafe.utils.preprocessing import SlidingWindow
from spafe.utils.vis import show_features

import pywt

def extract_bfcc(y, sr):
    lfccs  = bfcc(y,
                fs=sr,
                pre_emph=1,
                pre_emph_coeff=0.97,
                window=SlidingWindow(0.03, 0.015, "hamming"),
                nfilts=1024,
                nfft=2048,
                low_freq=0,
                high_freq=8000,
                normalize="mvn")
    return lfccs

def extract_cqcc(y, sr):
    cqccs  = cqcc(y,
                  fs=sr,
                  pre_emph=1,
                  pre_emph_coeff=0.97,
                  window=SlidingWindow(0.03, 0.015, "hamming"),
                  nfft=2048,
                  low_freq=0,
                  high_freq=8000,
                  normalize="mvn")

    return cqccs

def extract_lpc(y, sr):
    lpccs = lpcc(y,
             fs=sr,
             pre_emph=0,
             pre_emph_coeff=0.97,
             window=SlidingWindow(0.03, 0.015, "hamming"))
    return lpccs

def extract_mfcc(y, sr):
    mfccs  = mfcc(y,
              fs=sr,
              pre_emph=1,
              pre_emph_coeff=0.97,
              window=SlidingWindow(0.03, 0.015, "hamming"),
              nfilts=1024,
              nfft=2048,
              low_freq=0,
              high_freq=8000,
              normalize="mvn")
    return mfccs

def extract_mel(y, sr):
    mels = mel_spectrogram(y,
              fs=sr,
              pre_emph=1,
              pre_emph_coeff=0.97,
              window=SlidingWindow(0.03, 0.015, "hamming"),
              nfilts=1024,
              nfft=2048,
              low_freq=0,
              high_freq=8000)
    return mels

def extract_ssqcwt(y, sr = 16000):
    Twxo, Wxo, *_ = ssq_cwt(y, wavelet="morlet")
    return Wxo

def extract_cwt(y, sr = 16000):
    wavelet = 'morl' # wavelet type: morlet

    # scales for morlet wavelet
    widths = np.arange(1, 301, 1)
    # sampling period, timestep difference
    dt = 1/16000
    
    frequencies = pywt.scale2frequency(wavelet, widths) / dt # Get frequencies corresponding to scales
    wavelet_coeffs, freqs = pywt.cwt(y, widths, wavelet = wavelet, sampling_period=dt)
    # print("Shape of wavelet transform: ", wavelet_coeffs.shape)
    
    return wavelet_coeffs

def extract_cwt_example():
    fpath = "CON_T_0010584.wav"
    fs, sig = read(fpath)
    y = sig
    sr = fs
    wavelet = 'morl' # wavelet type: morlet
    widths = np.arange(10,90) # scales for morlet wavelet
    widths = np.concatenate((np.linspace(1,10,91), widths, np.linspace(90,100,5)))
    dt = 1/sr # timestep difference

    frequencies = pywt.scale2frequency(wavelet, widths) / dt # Get frequencies corresponding to scales
    wavelet_coeffs, freqs = pywt.cwt(y, widths, wavelet = wavelet, sampling_period=dt)
    print("Shape of wavelet transform: ", wavelet_coeffs.shape)
    

def extract_ssq_cwt(y):
    Twxo, Wxo, *_ = ssq_cwt(y, wavelet="morlet", mu=0)
    return Wxo

def extract_ssq_cwt_example():
    # read audio
    fpath = "CON_T_0010584.wav"
    fs, sig = read(fpath)
    print("sig shape: ", sig.shape)
    # compute ssq_cwt
    Twxo, Wxo, *_ = ssq_cwt(sig)
    print("Wxo shape: ", Wxo.shape)
    

def extract_lfcc(y, sr):
    lfccs  = lfcc(y,
                fs=sr,
                pre_emph=1,
                pre_emph_coeff=0.97,
                window=SlidingWindow(0.03, 0.015, "hamming"),
                nfilts=128,
                nfft=2048,
                low_freq=0,
                high_freq=8000,
                normalize="mvn")
    return lfccs

def extract_lfcc_example():

    # read audio
    fpath = "CON_T_0010584.wav"
    fs, sig = read(fpath)
    print("sig shape: ", sig.shape)
    # compute lfccs
    lfccs  = lfcc(sig,
                fs=fs,
                pre_emph=1,
                pre_emph_coeff=0.97,
                window=SlidingWindow(0.03, 0.015, "hamming"),
                nfilts=128,
                nfft=2048,
                low_freq=0,
                high_freq=8000,
                normalize="mvn")
    print("lfccs shape: ", lfccs.shape)
    print(np.amin(lfccs))
    print(np.amax(lfccs))


def extract_spectrogram(y, sr):
    pass

def extract_lpcs(y, sr):
    # compute lpcs
    lpcs, _ = lpc(sig=y,
                fs=sr,
                pre_emph=0,
                pre_emph_coeff=0.97,
                window=SlidingWindow(0.030, 0.015, "hamming"))
    return lpcs

def extract_lpcs_example():
    # read audio
    fpath = "CON_T_0010584.wav"
    fs, sig = read(fpath)
    print("sig shape: ", sig.shape)
    # compute lfccs
    lpcs, _ = lpc(sig,
                fs=fs,
                pre_emph=0,
                pre_emph_coeff=0.97,
                window=SlidingWindow(0.030, 0.015, "hamming"))
    print("lpcs shape: ", lpcs.shape)
    print(np.amin(lpcs))
    print(np.amax(lpcs))


def pad_to_dense_1d(M):
    """Appends the minimal required amount of zeroes at the end of each 
     array in the jagged array `M`, such that `M` looses its jagedness."""

    maxlen = max(len(r) for r in M)

    Z = np.zeros((len(M), maxlen))
    for enu, row in enumerate(M):
        Z[enu, :len(row)] += row 
    return Z

# def pad_to_dense_2d(jagged_array):
#     # Find the maximum number of rows among all 2D arrays in jagged_array
#     # For different datasets, we can fix the `max_num_rows` to a constant
#     max_num_rows = max(arr.shape[0] for arr in jagged_array)
#     num_columns = jagged_array[0].shape[1]  # Number of columns in each 2D array
#     print("max_num_rows: ", max_num_rows)
#     print("num_columns: ", num_columns)
#     # Create a new 2D array with dimensions (len(jagged_array), max_num_rows, num_columns)
#     padded_array = np.zeros((len(jagged_array), max_num_rows, num_columns))

#     # Copy the elements from each 2D array in jagged_array to the corresponding row in padded_array
#     for i, arr in enumerate(jagged_array):
#         padded_array[i, :arr.shape[0], :] = arr

#     return padded_array   

def pad_to_dense_2d(jagged_array):
    # Find the maximum number of columns among all 2D arrays in jagged_array
    max_num_columns = max(arr.shape[1] for arr in jagged_array)
    num_rows = jagged_array[0].shape[0]  # Number of rows in each 2D array

    # Create a new 2D array with dimensions (len(jagged_array), num_rows, max_num_columns)
    padded_array = np.zeros((len(jagged_array), num_rows, max_num_columns))

    # Copy the elements from each 2D array in jagged_array to the corresponding column in padded_array
    for i, arr in enumerate(jagged_array):
        padded_array[i, :, :arr.shape[1]] = arr

    return padded_array   

def normalize_dataset(dataset):
    # Calculate mean and standard deviation for the entire dataset
    mean = torch.mean(dataset)
    std = torch.std(dataset)

    # Define the normalization transform
    normalize = transforms.Normalize(mean=mean, std=std)

    # Create a tensor of ones with the same shape as the dataset
    ones = torch.ones_like(dataset)

    # Apply the normalization transform to the ones tensor to get the scaling factor
    scaling_factor = normalize(ones)

    # Normalize the dataset using broadcasting
    normalized_dataset = (dataset - mean) / scaling_factor

    return normalized_dataset


# extract_lfcc_example()
# extract_ssq_cwt_example()
# extract_lpcs_example()
# extract_cwt_example()