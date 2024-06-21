import soundfile as sf
import pandas as pd
import numpy as np

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.n_features = 8
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.n_features, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(self.n_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.n_features, 2 * self.n_features, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(2 * self.n_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * self.n_features, 4 * self.n_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * self.n_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * self.n_features, 4 * self.n_features, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(4 * self.n_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.estimation_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 120),
            nn.ReLU(),
            nn.Linear(120, 1),
            nn.ReLU())
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.estimation_layer(x)

        return x


class DatasetLoader(Dataset):
    def __init__(self, csv_path) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.frame_size=512
        self.hop_size = 256

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        signal, _ = librosa.load(str(self.df.loc[index]["ActorID"]), sr=16000)
        label = float(self.df.loc[index]["Age"])
        label = np.array(label, dtype=np.float32)

        stft = librosa.stft(signal, hop_length=self.hop_size, n_fft=self.frame_size, win_length=self.frame_size)
        stft = stft / np.max(np.abs(stft))
        stft = np.abs(stft) ** 2
        stft = 20 * np.log10(stft + 1e-8)
        stft = np.clip(stft, -80, 0)
        stft = np.expand_dims(stft, 0)
        stft = (stft / 40) + 1

        return stft, label

    
def train():

    BATCH_SIZE = 32 
    EPOCHS = 10
    LR = 1e-3
    BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"model running on device: {DEVICE}\n")
    train = DatasetLoader("train.csv")
    test = DatasetLoader("test.csv")

    print(f"number of training samples: {len(train)}")
    print(f"number of testing samples: {len(test)}")
    
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

    model = Model()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = 9999
    saved_epoch = -1
    for i in range(EPOCHS):
        print(f"\n+++++++++++++RUNNING EPOCH: {i} ++++++++++++++++")
        print("=============TRAINING===============")

        model.train().to(device=DEVICE).float()
        for x, y in tqdm(train_loader, bar_format=BAR_FORMAT):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(x)
            pred = torch.squeeze(pred)
            
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            
            mse = float(loss)
            mae = (y - pred).abs().mean()

        print(f"MSE TRAIN: =============={mse}")
        print(f"MAE TRAIN: =============={mae}\n")

        print("=============TESTING===============")
        model.eval().to(device=DEVICE)
        for x, y in tqdm(test_loader, bar_format=BAR_FORMAT):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            with torch.no_grad():
                pred = model(x)
                pred = torch.squeeze(pred)
                loss = loss_fn(pred, y)
                mse = float(loss)
                mae = (y - pred).abs().mean()
        
        print(f"MSE TEST: =============={mse}")
        print(f"MAE TEST: =============={mae}\n")
        
        if mae < history:
            history = mae
            saved_epoch = i
            print(f"model with {mae} mae saved at epoch {i}")
            torch.save(model.state_dict(), "saved_model.pt")

        

    print(f"model with {history} mae saved at epoch {saved_epoch}") 


if __name__ == "__main__":

    train()

