import glob
import librosa
import numpy as np

def wavToMFCC(file_path, sr = 16000, n_fft = 480, hop_length = 480, n_mfcc = 32):
    mfccimages = []
    for i in file_path:
        signal, sr = librosa.load(i, sr = sr)
        MFCCs = librosa.feature.mfcc(signal, sr = sr, \
            n_fft = n_fft, hop_length = hop_length, n_mfcc = n_mfcc)
        MFCCs = MFCCs[:, :, np.newaxis]
        mfccimages.append(MFCCs)

    return mfccimages
