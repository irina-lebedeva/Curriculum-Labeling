import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
from torch.utils.data import Dataset

sys.path.append('../')


class SCUTFBP5500Dataset(Dataset):

    """
    SCUT-FBP5500 dataset
    """

    def __init__(self, train=True, transform=None):
        split_train = '/home/ubuntu/SCUT5500/train_test_files/split_of_60%training and 40%testing/train.txt'
        split_test =  '/home/ubuntu/SCUT5500/train_test_files/split_of_60%training and 40%testing/test.txt'
        if train:
            self.face_img = pd.read_csv(split_train, sep=' ',names=['image','score'], header=None).iloc[:, 0].tolist()
            self.face_score = pd.read_csv(split_train,sep=' ',names=['image','score'], header=None).iloc[:,1].astype(np.integer).tolist()
            self.face_score = [x - 1 for x in self.face_score]
        else:
            self.face_img = pd.read_csv(split_test,sep=' ',names=['image','score'], header=None).iloc[:, 0].tolist()
            self.face_score = pd.read_csv(split_test, sep=' ', names=['image','score'], header=None).iloc[:, 1].astype(np.integer).tolist()
            self.face_score = [x - 1 for x in self.face_score]
        self.transform = transform

    def __len__(self):
        return len(self.face_img)

    def __getitem__(self, index):
        #image = io.imread('/home/ubuntu/SCUT5500/Images/'+str(self.face_img[index]))
        image=Image.open('/home/ubuntu/SCUT5500/Images/'+str(self.face_img[index]))
        score = self.face_score[index]
       # sample = {'image': image, 'score': score, 'class': round(score) - 1, 'filename': self.face_img[index]}

        
        if self.transform:
           image = self.transform(image)
        sample = image,  score
        return sample

class ECUSTDataset(Dataset):
    """
    ECUST dataset
    """

    def __init__(self, train=True, transform=None):
        if train:
            self.face_img = pd.read_csv('/home/ubuntu/train.txt',
                                        sep=' ', error_bad_lines=False, header=None).iloc[:, 0].tolist()
            self.face_score = pd.read_csv('/home/ubuntu/train.txt',
                                          sep=' ', header=None).iloc[:, 1].astype(np.float).tolist()
        else:
            self.face_img = pd.read_csv('/home/ubuntu/test.txt',
                                        sep=' ', error_bad_lines=False, header=None).iloc[:, 0].tolist()
            self.face_score = pd.read_csv('/home/ubuntu/test.txt',
                                          sep=' ', header=None).iloc[:, 1].astype(np.float).tolist()

        self.transform = transform

    def __len__(self):
        return len(self.face_img)

    def __getitem__(self, idx):
        image = io.imread(self.face_img[idx])
        score = self.face_score[idx]
        sample = {'image': image, 'score': score, 'class': round(score) - 1, 'filename': self.face_img[idx]}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample

    
    
