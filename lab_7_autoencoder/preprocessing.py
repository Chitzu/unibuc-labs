import pandas as pd
import os
import numpy as np
import soundfile

df = pd.DataFrame()

FIXED_LENGTH = 2 * 16000
data_path = "datasets/microsoft_dataset/clean_train"

train = os.listdir(data_path)

for idx, x in enumerate(train):
    signal, sr = soundfile.read(data_path + "/" + x)
    times = int(len(signal) / sr / 2)

    for i in range(times):
        new_signal = signal[i * FIXED_LENGTH: (i + 1) * FIXED_LENGTH]
        soundfile.write(f"datasets/microsoft/audio/audio_{idx}_{i}.wav", new_signal, sr)
        
files = os.listdir("datasets/microsoft/audio")
files = ["datasets/microsoft/audio/" + x for x in files]
df["file_name"] = np.array(files)[:int(0.2 * len(files))]

df.to_csv("test_files_microsoft.csv", header=False, index=False)