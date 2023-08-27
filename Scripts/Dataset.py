"""## Building dataset class"""

from torch.utils.data import Dataset
import torchaudio
import torch
import pandas as pd

class WakeWordDataset(Dataset):
  def __init__(self, csv, sample_rate, nbr_samples, device, transformations):
    #super.__init__()
    self.csv = pd.read_csv(csv)
    self.sample_rate = sample_rate
    self.nbr_samples = nbr_samples
    self.device = device
    self.transformations = transformations


  def __len__(self):
    return len(self.csv)

  def __getitem__(self, i):
    audio, sr = torchaudio.load(self.csv.iloc[i, 0])
    label = self.csv.iloc[i, 1]
    # Make sure that all audio's have the same sample_rate
    audio = self._resample_if_necessary(audio, sr)
    # Make sure every audio have the same channels .ie 1
    audio = self._mix_down_if_necessary(audio)
    # If the audio is longer than the wanted length, cut it down
    audio = self._cut_down_if_necessary(audio)
    # If the audio is shorter, fill it up
    audio = self._right_padding_if_necessary(audio)
    # Lastly create the Mel-Spectrogram
    audio = self.transformations(audio)
    return audio, label

  def _resample_if_necessary(self, audio, sr):
    if sr != self.sample_rate:
      resampler = torchaudio.transforms.Resample(sr, self.sample_rate).to(self.device)
      audio = resampler(audio)
    return audio

  def _mix_down_if_necessary(self, audio):
    if audio.shape[0] > 1:
      audio = torch.mean(audio, dim=0, keepdim=True)
    return audio

  def _cut_down_if_necessary(self, audio):
    if audio.shape[1] > self.nbr_samples:
      audio = audio[:, :self.nbr_samples]
    return audio

  def _right_padding_if_necessary(self, audio):
    if audio.shape[1] < self.nbr_samples:
      left_padding = (0, self.nbr_samples - audio.shape[1])
      audio = torch.nn.functional.pad(audio, left_padding)
    return audio
  
if __name__ == "__main__":

  if torch.cuda.is_available():
    device = "cuda"
  else:
    device = "cpu"

  SAMPLE_RATE= 48000
  NBR_SAMPLES = 240000

  print(f"device is {device}")

  mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                                        n_fft=1024,
                                                        hop_length=512, # half the frame_size n_fft
                                                        n_mels=64)

  data = WakeWordDataset(csv = "/content/train_data_wakeword.csv",
                      sample_rate = SAMPLE_RATE,
                      nbr_samples = NBR_SAMPLES,
                      device = device,
                      transformations = mel_spectrogram)

  audio, label = data[0]

  audio.shape