import os
import glob
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from wavToMFCC import wavToMFCC

class sound_dataset(Dataset):
    def __init__(self, path, img_size):
        self.path = path
        self.img_size = img_size

        self.audiodata, self.labels = self.scan(path)
        self.mfccdataset = wavToMFCC(self.audiodata, sr = 16000, \
            n_fft = 480, hop_length = 480, n_mfcc = 32)

    def __len__(self):
        return len(self.mfccdataset)

    def __getitem__(self, index):
        image = self.mfccdataset[index]
        label = self.labels[index]
        
        image = self.transform(image)
        
        return image, label

    def scan(self, path):
        dataset_type = ['snoring', 'nonsnoring']
        file_path = []
        label = []
        for type in dataset_type:
            file_list = sorted(glob.glob(path + '/' + type + '/*.wav'))
            file_path.extend(file_list)

            if type == 'snoring':
                for i in range(500):
                    label.append(1)
            else:
                for i in range(500):
                    label.append(0)

        return file_path, label

    def transform(self, img):
        img = np.resize(img, (self.img_size, self.img_size, 1))
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)
        return img

def make_dataloader(path, img_size, train_test_ratio, valid_ratio, batch_size):
    dataset = sound_dataset(path = path, img_size = img_size)
    test_length = int(len(dataset) * train_test_ratio)
    valid_length = int((len(dataset) - test_length) * valid_ratio)
    train_data, valid_data, test_data = torch.utils.data.random_split(dataset, \
        [len(dataset) - valid_length - test_length, valid_length, test_length])

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)

    return train_loader, valid_loader, test_loader

