import torchaudio
from torchaudio_augmentations import *
import os
import torch
import argparse
from tqdm import tqdm



def rename_from_wave_to_wav(path_dir):
    # Specify the directory path
    directory_path = path_dir

    # List all files in the specified directory
    files = os.listdir(directory_path)

    # Iterate through files and rename .wave files to .wav
    for old_filename in files:
        if old_filename.endswith('.wave'):
            old_file_path = os.path.join(directory_path, old_filename)
            new_filename = old_filename.replace('.wave', '.wav')
            new_file_path = os.path.join(directory_path, new_filename)
            os.rename(old_file_path, new_file_path)

def augmentation_process(directory, probabilities, nbr_generated_samples, save_directory):
    for i in tqdm(range(probabilities.shape[0])):
        for file in tqdm(os.listdir(directory)):
            audio, sr = torchaudio.load(os.path.join(directory, file))
            num_samples = sr * 5
            transforms = [
            RandomApply([PolarityInversion()], p=probabilities[i][0]),
            RandomApply([Noise(min_snr=0.001, max_snr=0.01)], p=probabilities[i][1]),
            RandomApply([Gain()], p=probabilities[i][2]),
            RandomApply([PitchShift(n_samples=num_samples,sample_rate=sr)], p=probabilities[i][3]),
            RandomApply([Reverb(sample_rate=sr)], p=probabilities[i][4])
            ]

            transform = ComposeMany(transforms=transforms, num_augmented_samples=nbr_generated_samples)
            transformed_audio = transform(audio)

            j = 0
            for f in transformed_audio:
                torchaudio.save(os.path.join(save_directory,f"config{i}_trans{j}_{file}"), f, sr)
                j += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-dir", "--data_directory",  required=True,
                        help="The directory of the data that you want augmented.")
    parser.add_argument("-smp", "--nbr_of_aug_samples", type=int, default=5, required=True,
                        help="number of generated augmented samples.")
    parser.add_argument("-sdir", "--save_directory", required=True,
                        help="Saving directory.")
    args = parser.parse_args()

    directory = args.data_directory
    #directory2 = "wakeword_test_data"
    rename_from_wave_to_wav(directory)
    #rename_from_wave_to_wav(directory2)

    prob = torch.rand((args.nbr_of_aug_samples,5)) *0.8
    prob = prob + 0.2
    prob = torch.round(prob * 10) / 10

    augmentation_process(directory=directory, 
                         probabilities=prob, 
                         nbr_generated_samples=args.nbr_of_aug_samples,
                         save_directory=args.save_directory)
