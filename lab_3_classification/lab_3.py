import soundfile as sf
import pandas as pd
import numpy as np

from models.features_svm import features_svm
from models.fft_svm import fft_svm
from models.wave_svm import wave_svm
from models.stft_cnn import stft_cnn

def load_data(csv_path):

    df = pd.read_csv(csv_path)

    tensors = []
    labels = []
    for i, df in enumerate(df.iterrows()):
        
        file = str(df[1]["name"])
        label = int(df[1]["gender"])
        sig, sr = sf.read(file)

        tensors.append(sig)
        labels.append(label)

    tensors = np.stack(tensors, axis=0, dtype=np.float32)
    labels = np.stack(labels, axis=0, dtype=np.float32)

    print(f"input shape for {csv_path[:-4]} data: {tensors.shape}")
    return tensors, labels      

if __name__ == "__main__":

    x_train, y_train = load_data("train.csv")
    x_test, y_test = load_data("test.csv")

    print("Running SVM on waveform")
    wave_svm(x_train, y_train, x_test, y_test)
    
    print("\nRunning SVM on FFT")
    fft_svm(x_train, y_train, x_test, y_test)
    
    print("\nRunning SVM on mean, standard deviation and power of the waveform")
    features_svm(x_train, y_train, x_test, y_test)
    
    print("\nRunning CNN on the spectogram of the signal")
    stft_cnn()

