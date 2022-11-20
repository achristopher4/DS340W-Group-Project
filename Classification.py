# Classification Model
## Date: 11/20/2022


import SteelDataset as sd
import os
import pandas as pd
from time import gmtime, strftime

import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential

class Classification(object):
    def __init__(self, train_num, num_segmented, fold) :
        self.train_num = train_num
        self.num_segmented = num_segmented
        self.train_datadir = './datasets/STEEL/train_images'
        self.data =  sd.STEELDataAcquisition('TRAIN', train_num, num_segmented)
        self.result_path = ['./result', self.generate_result_name()]
        ##
        # Appears to be testing model on testing data
        self.validation = sd.STEELDataAcquisition('TEST', train_num, num_segmented)
        ##
        self.test_data = None
        self.img_height, self.img_width = (Image.open('./datasets/STEEL/train_images/0a1cade03.jpg')).size
        self.class_names = list(self.data.training.keys()) 
        self.model = self.train_model()
        self.eval_model = self.eval()
    
    def generate_result_name(self):
        models = (os.listdir('./results'))
        baseName = 'STEEL_CLASSSIFICATION'
        if len(models) == 0:
            postfix = 0
        else:
            postfix = 1
            path = f'./results/{baseName}_{postfix}'
            while os.path.exists(path):
                postfix += 1
                path += f'./results/{baseName}_{postfix}'
        name = f'{baseName}_{postfix}'
        print('\n', '-'*50, '\n', 'Model Name:\n\t' ,name, '\n', '-'*50, '\n')
        return name
    
    def train_model(self):
        #img = PIL.Image.open(str( './datasets/STEEL/train_images/0a1cade03.jpg' ))
        #img.show()

        if len(os.listdir('./STEEL_MODEL')) > 0:
            return tf.keras.models.load_model('./STEEL_MODEL')

        data_dir = './STEEL_TRAINING'
        batch_size = 32
        img_height, img_width = (Image.open('./datasets/STEEL/train_images/0a1cade03.jpg')).size
        self.img_height = img_height
        self.img_width = img_width

        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split = 0.2, 
            subset = "training",
            seed = 123,
            image_size =(img_height, img_width),
            batch_size = batch_size
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split = 0.2,
            subset = "validation",
            seed = 123, 
            image_size = (img_height, img_width),
            batch_size = batch_size
        )

        class_names = list(self.data.training.keys())
        self.class_names = class_names

        AUTOTUNE = tf.data.AUTOTUNE
        
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)

        num_classes = len(class_names)

        normalization_layer = layers.Rescaling(1./255)

        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        first_image = image_batch[0]

        #print(np.min(first_image), np.max(first_image))

        num_classes = len(class_names)
        model = Sequential([
            layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        print(model.summary())

        epochs=10
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal",
                                input_shape=(img_height,
                                            img_width,
                                            3)),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
            ]
        )

        plt.figure(figsize=(10, 10))
        for images, _ in train_ds.take(1):
            for i in range(9):
                augmented_images = data_augmentation(images)
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(augmented_images[0].numpy().astype("uint8"))
                plt.axis("off")
        
        model = Sequential([
            data_augmentation,
            layers.Rescaling(1./255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, name="outputs")
        ])

        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        
        model.summary()

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

        model.save('./STEEL_MODEL')
        return model
        
    def test_results(self):
        steel_path = './datasets/STEEL/test_images/'
        test_imgs = os.listdir(steel_path)
        pred_score = pd.DataFrame(columns = ['pred_value', 'filename', 'pred_confidence'])
        count = 0
        for t_img in test_imgs:
            img = tf.keras.utils.load_img(
                steel_path + t_img, target_size=(self.img_height, self.img_width)
            )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = self.model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            if t_img == '00a0b7730.jpg':
                print(
                    "This image most likely belongs to {} with a {:.2f} percent confidence."
                    .format(self.class_names[np.argmax(score)], 100 * np.max(score))
                )

            pred_value = self.class_names[np.argmax(score)]

            pred_score.loc[count] = [pred_value, t_img, 100 * np.max(score)]
            count += 1
        pred_score.to_csv(f'RESULTS/Test_{strftime("%Y-%m-%d_%H:%M:%S", gmtime())}.csv')
        return pred_score

    def eval(self):
        print('\n', '-'*50, '\n', 'Evaluation\n', '\n', '-'*50, '\n')
        if len(os.listdir('./RESULTS')) == 0:
            results_df = self.test_results()
        else:
            csv_filename = os.listdir('./RESULTS')[0]
            results_df = pd.read_csv('./RESULTS/' + csv_filename)
        test_validation = sd.STEELDataAcquisition('TEST',  self.train_num, self.num_segmented)
        positve_set = set([row[-1].split('/')[-1] +'.jpg' for row in test_validation.training['pos']])
        negative_set = set([row[-1].split('/')[-1] + 'jpg' for row in test_validation.training['neg']])
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        #print(results_df.pred_value.unique())
        for index, row in results_df.iterrows():
            r_img = row['filename']
            r_pred = row['pred_value']
            if r_img in positve_set and r_pred == 'pos':
                true_positive += 1
            elif r_img in positve_set and r_pred == 'neg':
                false_positive += 1
            elif r_img in negative_set and r_pred == 'neg':
                true_negative += 1
            elif r_img in negative_set and r_pred == 'pos':
                false_negative += 1
        
        print('\nTesting Set Accuarcy:')
        print(f'\tTP: {true_positive}\tTN: {true_negative}\n\tFP: {false_positive}\tFN: {false_negative}')
        #accuracy = (true_negative + true_positive) / (true_negative + true_positive + false_negative + false_positive)
        #print(f'\tAccuracy: {accuracy}')

        


