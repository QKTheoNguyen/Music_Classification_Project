from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import torchaudio
import os

class_mapping = {
    'blues': 0,
    'classical': 1,
    'country': 2,
    'disco': 3,
    'hiphop': 4,
    'jazz': 5,
    'metal': 6,
    'pop': 7,
    'reggae': 8,
    'rock': 9
}

class MusicDataset(Dataset):
    def __init__(self, config, metadata_file, data_dir, transformation, device, random_sample):
        self.metadata = pd.read_csv(metadata_file)
        self.config = config
        self.data_dir = data_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.random_sample = random_sample

        self.n_fft = self.config["n_fft"]
        self.hop_length = self.config["hop_length"]
        self.target_sr = self.config["target_sr"]
        self.full_audio_length = self.config["full_audio_length"]
        self.transformation_type = self.config["transform"]

        
        if self.config["n_frames"] is not None and self.config["duration"] is not None:
            raise ValueError("You can't specify both n_frames and duration")
        elif self.config["n_frames"] is not None:
            self.n_frames = self.config["n_frames"]
            self.n_samples = (self.n_frames - 1) * self.hop_length + self.n_fft
            self.duration = self.n_samples / self.target_sr
        elif self.config["duration"] is not None:
            self.duration = self.config["duration"]
            self.n_samples = self.duration * self.target_sr
            self.n_frames = 1 + (self.n_samples - self.n_fft) // self.hop_length

        print(f'Loading audio on length {self.n_samples}, duration {"{:.2f}".format(self.duration)}s, sample rate {self.target_sr}Hz')

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        label = self.metadata.iloc[index].label
        label_int = class_mapping[label]
        audio_path = os.path.join(self.data_dir, label, self.metadata.iloc[index].filename)
        signal, sr = torchaudio.load(audio_path, normalize=True)
        signal = signal.to(self.device)
        if self.full_audio_length:
            onset = self.metadata.iloc[index].onset
            offset = self.metadata.iloc[index].offset
        if sr != self.target_sr:
            signal = torchaudio.transforms.Resample(sr, self.target_sr)(signal)

        if signal.size(1) != self.n_samples and not self.full_audio_length:
            signal = self._reshape_signal(signal, random_sample=self.random_sample)

        if self.full_audio_length:
            signal = signal[:, onset:offset]
            if signal.size(1) < self.n_samples:
                signal = torch.nn.functional.pad(signal, (0, self.n_samples - signal.size(1)))

        # normalize the signal (peak normalization)
        mean = signal.mean(dim=1, keepdim=True)
        peak = torch.max(torch.max(signal), torch.min(signal))
        signal = (signal - mean) / (1e-10 + peak)

        signal = self.transformation(signal)

        if self.transformation_type == "mel_spectrogram":

            ## normalize the output

            ## normal distribution
            mean = signal.mean(dim=2, keepdim=True)
            stdev = signal.std(dim=2, keepdim=True)
            signal = (signal - mean) / (stdev + 1e-10) 

            ## log normalization
            # signal = torch.log(signal)

        return signal, label_int
    
    def _reshape_signal(self, signal, random_sample):
        len_sig = signal.size(1)
        if len_sig < self.n_samples:
            signal = torch.nn.functional.pad(signal, (0, self.n_samples - len_sig))
        else:
            if random_sample:
                max_offset = len_sig - self.n_samples
                offset = torch.randint(0, max_offset, (1,)).item()
                signal = signal[:, offset:offset + self.n_samples]
            else:
                signal = signal[:, :self.n_samples]
        return signal


if __name__ == "__main__":

    import yaml

    def load_config(config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    split = 'test'
    metadata_file = f'data/metadata_30_sec_{split}.csv'
    data_dir = "data/genres_original"
    config_path = "config/config.yaml"
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_sr = config["target_sr"]
    n_fft = config["n_fft"]
    hop_length = config["hop_length"]
    n_mels = config["n_mels"]
    transform = config["transform"]


    if transform == "mel_spectrogram":

        transformation = torchaudio.transforms.MelSpectrogram(sample_rate=target_sr,
                                                              n_fft=n_fft,
                                                              hop_length=hop_length,
                                                              n_mels=n_mels,
                                                              normalized=False,
                                                              center=False)
    
    elif transform == "mfcc":
        transformation = torchaudio.transforms.MFCC(sample_rate=target_sr,
                                                    n_mfcc=n_mels,
                                                    melkwargs={"n_fft": n_fft, 
                                                                "hop_length": hop_length, 
                                                                "n_mels": n_mels,
                                                                "normalized": False,
                                                                "center": False})

    dataset = MusicDataset(config, metadata_file, data_dir, transformation, device, random_sample=True)
    # print(f'dataset length : {len(dataset)}')

    # signal, label = dataset[0]
    # print(f'sample shape : {signal.shape}, label : {label}')

    train_loader = DataLoader(dataset=dataset, 
                            batch_size=5, 
                            shuffle=False)
    
    (data, target) = next(iter(train_loader))

    print(f"Data shape : {data.size()}")
    print(f"Target shape : {target.size()}")
    print(f"Target : {target}")
    print(f'Data min : {data.min()}, max : {data.max()}')