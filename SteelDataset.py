# Retireve and Transform Steel dataset into array
## Date: 11/20/2022

import os
import pickle
import numpy as np
import pandas as pd

class STEELDataAcquisition(object):
    def __init__(self, kind, train_num, num_segmented):
        self.basePath = './datasets/STEEL/'
        self.train_num = train_num
        self.num_segmented = num_segmented
        self.training = {'pos': [], 'neg': []}
        if kind == 'TRAIN' or 'TEST':
            self.kind = kind
            if kind == 'TRAIN':
                self.imagePath = self.basePath + 'train_images'
                self.folder = 'train_images'
            else:
                self.imagePath = self.basePath + 'test_images'
                self.folder = 'test_images'
            self.annotated_data = self.annotatedData()
            self.splits = self.read_splits()
            self.data = self.retrieveData()
            self.num_pos = len(self.training['pos'])
            self.num_neg = len(self.training['neg'])
        else:
            print("Invalid. Choose 'TRAIN' or 'TEST'.")
            return None

    def annotatedData(self):
        fn = os.path.join(self.basePath, "train.csv")
        arr = np.array(pd.read_csv(fn), dtype=np.object)
        annotations_dict = {}
        for sample, _, rle in arr:
            img_name = sample[:-4]
            annotations_dict[img_name] = rle
        return annotations_dict

    def read_splits(self):
        ## train_num: Number of positive training samples for STEEL.
        ## num_segmented: Number of segmented positive  samples.
        steel_split_path = './splits/'
        fn = f"STEEL/split_{self.train_num}_{self.num_segmented}.pyb"
        with open(f"./splits/{fn}", "rb") as f:
            train_samples, test_samples, validation_samples = pickle.load(f)
            if self.kind == 'TRAIN':
                return train_samples
            elif self.kind == 'TEST':
                return test_samples
            elif self.kind == 'VAL':
                return validation_samples
            else:
                raise Exception('Unknown')
        return fn
    
    def retrieveData(self):
        for sample, is_segmented in self.splits:
            img_name = f"{sample}.jpg"
            img_path = os.path.join(self.basePath, self.folder, img_name)
            if sample in self.annotated_data:
                rle = list(map(int, self.annotated_data[sample].split(" ")))
                self.training['pos'].append((None, None, None, is_segmented, img_path, rle, sample))
            else:
                self.training['neg'].append((None, None, None, True, img_path, None, sample))






