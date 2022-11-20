# Main file for collecting data that will be used in model

import SteelDataset as sd
import shutil

train_num = 3000
num_segmented = 0
training_data = sd.STEELDataAcquisition('TRAIN', train_num, num_segmented)

train_pos_des = './STEEL_TRAINING/POSITIVE'
train_neg_des = './STEEL_TRAINING/NEGATIVE'

## Create positve samples train folder
for pos_sample in training_data.training['pos']:
    ## Copy and place into POSITIVE folder
    src = pos_sample[4]
    shutil.copy(src, train_pos_des)

## Create negative smaples train folder
for neg_sample in training_data.training['neg']:
    ## Copy and place into NEGATIVE folder
    src = neg_sample[4]
    shutil.copy(src, train_neg_des)
