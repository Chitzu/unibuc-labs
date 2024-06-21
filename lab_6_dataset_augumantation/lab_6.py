import os.path
import librosa
import soundfile as sf
import torch.utils.data
import glob
import os
import numpy as np
import random
import pandas as pd
import json
import matplotlib.pyplot as plt

class SCDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode="train"):
        self.config = config
        self.data = []
        self.back_ground_noise = []
        self.training = True if mode == "train" else False
        self.FIXED_LENGTH = 16000
        self.HOP_SIZE = config['HOP_SIZE']
        self.FRAME_SIZE = config['FRAME_SIZE']
        self._process_dataset(mode)

    def _process_dataset(self, mode):
        classes_dirs = glob.glob(os.path.join(self.config['path'], "*/"))
        classes_dirs.sort()
        testing_list = list(pd.read_csv(self.config['testing_list'], header=None).to_numpy()[:, 0])   
        testing_list = [os.path.join(self.config['path'], x) for x in testing_list]

        for idx, class_path in enumerate(classes_dirs):
            if "_background_noise_" in class_path:
                self.back_ground_noise = glob.glob(os.path.join(class_path, "*.wav"))
                print(f"background noise: {self.back_ground_noise}")
                continue

            files = glob.glob(os.path.join(class_path, "*.wav"))

            if mode == "train":
                files = [x for x in files if x not in testing_list]

            elif mode == "test":
                files = [x for x in files if x in testing_list]
            
            self.data = self.data + files

    def __getitem__(self, index):
        raw_signal, fs = sf.read(self.data[index])
        raw_signal = self.get_fix_length(raw_signal)

        clean_raw_signal = raw_signal
        stft = librosa.stft(raw_signal, hop_length=self.HOP_SIZE, n_fft=self.FRAME_SIZE, win_length=self.FRAME_SIZE)
        stft = stft / np.max(np.abs(stft))
        stft = np.abs(stft)**2
        stft = 20*np.log10(stft + 1e-8) 
        stft = np.clip(stft, -80, 0)
        clean_stft = np.expand_dims(stft, 0)
       
        if random.uniform(0, 1) < self.config['augment_chance']:
            raw_signal = self.augment_signal(raw_signal, index)

        stft = librosa.stft(raw_signal, hop_length=self.HOP_SIZE, n_fft=self.FRAME_SIZE, win_length=self.FRAME_SIZE)
        stft = stft / np.max(np.abs(stft))
        stft = np.abs(stft)**2
        stft = 20*np.log10(stft + 1e-8) 
        stft = np.clip(stft, -80, 0)
        augmented_stft = np.expand_dims(stft, 0)
        return clean_stft, clean_raw_signal, augmented_stft, raw_signal, fs
   
    def augment_signal(self, raw_signal, index):
       if self.config['noise_augment'] and self.training:
           if random.uniform(0, 1) < self.config['noise_augment_chance']:
               raw_signal = self.noise_augmentation(raw_signal)

       if self.config['time_augment'] and self.training:
           if random.uniform(0, 1) < self.config['time_augment_chance']:
               raw_signal = np.roll(raw_signal, random.randint(0, int(self.FIXED_LENGTH / 3)))

       if self.config['speed_augment'] and self.training:
           if random.uniform(0, 1) < self.config['speed_augment_chance']:
               raw_signal = self.speed_augment(raw_signal)
               
       if self.config['volume_augment'] and self.training:
           if random.uniform(0, 1) < self.config['volume_augment_chance']:
               raw_signal = self.volume_augment(raw_signal)
       return raw_signal
   
    def noise_augmentation(self, raw_signal):
        if random.uniform(0, 1) < self.config['gaussian_noise_augment_chance']:
            target_snr_db = np.random.randint(30, 70)
            sig_avg_watts = np.mean(raw_signal ** 2)
            sig_avg_db = 10 * np.log10(sig_avg_watts)
            noise_avg_db = sig_avg_db - target_snr_db
            noise_avg_watts = 10 ** (noise_avg_db / 10)
            mean_noise = 0
            noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), raw_signal.shape)
            raw_signal = raw_signal + noise
        else:
            noise, _ = sf.read(self.back_ground_noise[np.random.randint(0, len(self.back_ground_noise))])
            start_idx = np.random.randint(0, len(noise) - self.FIXED_LENGTH - 1)
            noise = noise[start_idx:start_idx + self.FIXED_LENGTH]  
            target_snr_db = np.random.randint(0, 60)
            snr_noise_coeff = (raw_signal ** 2).mean() / ((noise ** 2).mean() * np.power(10, target_snr_db / 20))
            noise = np.sqrt(snr_noise_coeff) * noise
            raw_signal = raw_signal + noise 
        return raw_signal
   
    def speed_augment(self, data):
        speed_factor = np.random.uniform(0.85, 1.15, 1)[0]
        data = librosa.effects.time_stretch(data, rate=speed_factor)
        return self.get_fix_length(data)

    def volume_augment(self, raw_signal, vol_range_db=6):
        vol_scale_factor = np.random.uniform(-vol_range_db, vol_range_db)
        raw_signal_scaled = raw_signal * (10 ** (vol_scale_factor / 20))
        return raw_signal_scaled
   
    def get_fix_length(self, raw_signal):
       if len(raw_signal) == self.FIXED_LENGTH:
           return raw_signal
       
       if len(raw_signal) < self.FIXED_LENGTH:
           new_signal, _ = sf.read(self.data[np.random.randint(0, len(self.data))])
           raw_signal = self.get_fix_length(np.concatenate((raw_signal, new_signal), -1))
           return raw_signal
       
       elif len(raw_signal) > self.FIXED_LENGTH and self.training is True:
           len_difference = np.abs(len(raw_signal) - self.FIXED_LENGTH)
           rand_idx = np.random.randint(0, len_difference)
           raw_signal = raw_signal[rand_idx:rand_idx+self.FIXED_LENGTH]

       return raw_signal
   
    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    f = open('config.json')
    config = json.load(f)
    HOP_SIZE = config['HOP_SIZE']
    FRAME_SIZE = config['FRAME_SIZE']
    dataset = SCDataset(config)
    print("dataset length: ",len(dataset))

    for idx, (stft_clean, signal_clean, stft_aug, signal_aug, sr) in enumerate(dataset):
        
        fig, axs = plt.subplots(4, figsize=(12,9))
        spec1 = librosa.display.specshow(np.squeeze(stft_clean), hop_length=HOP_SIZE, n_fft=FRAME_SIZE, win_length=FRAME_SIZE, 
                                        ax=axs[0], y_axis='log', x_axis='time', cmap='plasma')
        spec2 = librosa.display.specshow(np.squeeze(stft_aug), hop_length=HOP_SIZE, n_fft=FRAME_SIZE, win_length=FRAME_SIZE, 
                                        ax=axs[1], y_axis='log', x_axis='time', cmap='plasma')
        
        librosa.display.waveshow(signal_clean, sr=sr, ax=axs[2])
        librosa.display.waveshow(signal_aug, sr=sr, ax=axs[3])

        fig.colorbar(spec1, ax=axs[0], format="%+2.f")
        fig.colorbar(spec1, ax=axs[1], format="%+2.f")
        fig.savefig(f'spectrograms/spec{idx}.png')


