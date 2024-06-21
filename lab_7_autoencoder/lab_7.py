import torch
import librosa
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import glob
import os
import soundfile as sf
import json 
import scipy
import matplotlib.pyplot as plt
from networks.autoencoder import AE

class AEDataset(Dataset):
    def __init__(self, config, mode="train"):
        self.config = config
        self.data = []
        self.back_ground_noise = []
        self.training = True if mode == "train" else False
        self.FIXED_LENGTH = 2 * 16000
        self.HOP_SIZE = config['HOP_SIZE']
        self.FRAME_SIZE = config['FRAME_SIZE']
        self.max = []
        self._process_dataset(mode)

    def _process_dataset(self, mode):
        classes_dirs = glob.glob(os.path.join(self.config['path'], "*/"))
        classes_dirs.sort()
        testing_list = list(pd.read_csv(self.config['testing_list'], header=None).to_numpy()[:, 0])

        for idx, class_path in enumerate(classes_dirs):
            files = glob.glob(os.path.join(class_path, "*.wav"))

            if mode == "train":
                files = [x for x in files if x not in testing_list]

            elif mode == "test":
                files = [x for x in files if x in testing_list]
            
            self.data = self.data + files
            print(len(self.data), mode)

    def __getitem__(self, index):
        raw_signal, fs = sf.read(self.data[index])
        raw_signal = self.get_fix_length(raw_signal)

        stft = librosa.stft(raw_signal, hop_length=self.HOP_SIZE, n_fft=self.FRAME_SIZE, win_length=self.FRAME_SIZE)
        # self.max.append(np.max(stft))
        stft = stft / 118.
        clean_stft = np.stack((np.real(stft), np.imag(stft)), dtype=np.float32)

        noisy_raw_signal = self.noise_augmentation(raw_signal)

        stft = librosa.stft(noisy_raw_signal, hop_length=self.HOP_SIZE, n_fft=self.FRAME_SIZE, win_length=self.FRAME_SIZE)
        stft = stft / 118.
        noisy_stft = np.stack((np.real(stft), np.imag(stft)), dtype=np.float32)
        noisy_stft = noisy_stft[:,:-2,:]  ## remove 2 elements
        noisy_stft = np.concatenate((noisy_stft, np.zeros((2,255,4))), dtype=np.float32, axis=-1) ### add 4 zeros for dim issue
        return noisy_stft, clean_stft
    
    def noise_augmentation(self, raw_signal):
        target_snr_db = 10
        sig_avg_watts = np.mean(raw_signal ** 2)
        sig_avg_db = 10 * np.log10(sig_avg_watts + 1e-8)
        noise_avg_db = sig_avg_db - target_snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        mean_noise = 0
        noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), raw_signal.shape)   
        raw_signal = raw_signal + noise
        
        return raw_signal
    
    def get_fix_length(self, raw_signal):
       if len(raw_signal) == self.FIXED_LENGTH:
           return raw_signal
       
       if len(raw_signal) < self.FIXED_LENGTH:
           new_signal, _ = sf.read(self.data[np.random.randint(0, len(self.data))])
           raw_signal = self.get_fix_length(np.concatenate((raw_signal, new_signal), -1))
           return raw_signal
       
       elif len(raw_signal) > self.FIXED_LENGTH:
           len_difference = np.abs(len(raw_signal) - self.FIXED_LENGTH)
           rand_idx = np.random.randint(0, len_difference)
           raw_signal = raw_signal[rand_idx:rand_idx+self.FIXED_LENGTH]
       return raw_signal
    
    def __len__(self):
        return len(self.data)

def train():

    BATCH_SIZE = 1
    EPOCHS = 20
    LR = 1e-3
    BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    f = open('config.json')
    config = json.load(f)

    print(f"model running on device: {DEVICE}\n")
    train = AEDataset(config, mode="train")
    test = AEDataset(config, mode="test")

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

    model = AE()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = 9999
    saved_epoch = -1
    
    for i in range(EPOCHS):
        print(f"+++++++++++++RUNNING EPOCH: {i} ++++++++++++++++")
        print("=============TRAINING===============")

        mse_stats = []
        model.train().to(device=DEVICE).float()
        for x, y in tqdm(train_loader, bar_format=BAR_FORMAT):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(x)
            zeros = torch.zeros(pred.shape[0], pred.shape[1], 2 ,pred.shape[3]).to(DEVICE)
            pred = torch.concatenate((pred, zeros), dim=-2)
            pred = pred[:,:,:,:-4]
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            mse = float(loss)
            mae = (y - pred).abs().mean()
            mse_stats.append(mse)

        mse_stats = np.array(mse_stats)
        mse = np.mean(mse_stats)
        print(f"MSE TRAIN: =============={mse}")
        print(f"MAE TRAIN: =============={mae}\n")

        # maxim = np.array(train.max)
        # print(np.max(maxim))

        print("=============TESTING===============")
        mse_stats = []
        model.eval().to(device=DEVICE)
        for x, y in tqdm(test_loader, bar_format=BAR_FORMAT):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            with torch.no_grad():

                pred = model(x)
                zeros = torch.zeros(pred.shape[0], pred.shape[1], 2 ,pred.shape[3]).to(DEVICE)
                pred = torch.concatenate((pred, zeros), dim=-2)
                pred = pred[:,:,:,:-4]
                loss = loss_fn(pred, y)
                x = torch.concatenate((x, zeros), dim=-2)
                x = x[:,:,:,:-4]
                mse = float(loss)
                mae = (y - pred).abs().mean()
                mse_stats.append(mse)

        # maxim = np.array(test.max)
        # print(np.max(maxim))
        
        mse_stats = np.array(mse_stats)
        mse = np.mean(mse_stats)

        y = y.detach().cpu().numpy()[0,:,:,:]
        pred = pred.detach().cpu().numpy()[0,:,:,:]
        x = x.detach().cpu().numpy()[0,:,:,:]

        y = y[0] + y[1]*1j
        x = x[0] + x[1]*1j
        pred = pred[0] + pred[1]*1j

        fig, axs = plt.subplots(6, figsize=(13, 10))
        clean = librosa.istft(118. * y, hop_length=config["HOP_SIZE"], n_fft=config["FRAME_SIZE"], win_length=config["FRAME_SIZE"])
        noisy = librosa.istft(118. * x, hop_length=config["HOP_SIZE"], n_fft=config["FRAME_SIZE"], win_length=config["FRAME_SIZE"])
        enhanced = librosa.istft(118. * pred, hop_length=config["HOP_SIZE"], n_fft=config["FRAME_SIZE"], win_length=config["FRAME_SIZE"])

        librosa.display.waveshow(clean, sr=16000, ax=axs[0])
        librosa.display.waveshow(noisy, sr=16000, ax=axs[1])
        librosa.display.waveshow(enhanced, sr=16000, ax=axs[2])

        librosa.display.specshow(y, hop_length=config["HOP_SIZE"], n_fft=config["FRAME_SIZE"], win_length=config["FRAME_SIZE"],
                                ax=axs[3], y_axis='log', x_axis='time', cmap='plasma', sr=16000)
       
        librosa.display.specshow(x, hop_length=config["HOP_SIZE"], n_fft=config["FRAME_SIZE"], win_length=config["FRAME_SIZE"],
                                ax=axs[4], y_axis='log', x_axis='time', cmap='plasma', sr=16000)
        
        librosa.display.specshow(pred, hop_length=config["HOP_SIZE"], n_fft=config["FRAME_SIZE"], win_length=config["FRAME_SIZE"],
                                ax=axs[5], y_axis='log', x_axis='time', cmap='plasma', sr=16000)
        
        axs[0].set_title("Clean Signal")
        axs[1].set_title("Noisy Signal")
        axs[2].set_title("Ehanced Signal")
        axs[3].set_title("Clean Spectrogram")
        axs[4].set_title("Noisy Spectrogram")
        axs[5].set_title("Ehanced Spectrogram")

        
        if not os.path.exists(f"results/epoch_{i}"):
            os.mkdir(f"results/epoch_{i}")
        fig.savefig(f'results/epoch_{i}/waveform.png')

        scipy.io.wavfile.write(f"results/epoch_{i}/clean.wav", 16000, clean)
        scipy.io.wavfile.write(f"results/epoch_{i}/noisy.wav", 16000, noisy)
        scipy.io.wavfile.write(f"results/epoch_{i}/enhanced.wav", 16000, enhanced)

        print(f"MSE TEST: =============={mse}")
        print(f"MAE TEST: =============={mae}\n")

        if mse < history:
            history = mse
            saved_epoch = i

            torch.save(model.state_dict(), "saved_model.pt")

            print(f"model with {mse} mse saved at epoch {i}")

    print(f"model with {history} mae saved at epoch {saved_epoch}") 

if __name__ == "__main__":
    train()