print("Starting indo_fash.py ...")

# local tools and their requirements
import sys
sys.path.append("../") # used to allow import from utils folder
import utils.req_functions as rf
import matplotlib.pyplot as plt

# external imports
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)

# load images from file
def load_image():
    # defining filepaths as they appear on UCloud
    img_dir = os.path.join('..','..','..','431824')
    metadata = os.path.join('..','..','..','431824','images','metadata')

    train_data_dir = os.path.join(img_dir,'images','train')
    test_data_dir = os.path.join(img_dir,'images','test')
    val_data_dir = os.path.join(img_dir,'images','val')

    # reading metadata from .json to dataframes
    train_df = pd.read_json(os.path.join(metadata,'train_data.json'), lines=True)
    test_df = pd.read_json(os.path.join(metadata,'test_data.json'), lines=True)
    val_df = pd.read_json(os.path.join(metadata,'val_data.json'), lines=True)

    # using fractions of full dataset as it is quite large
    #train_df = train_df.sample(frac=0.1, random_state=42)
    #val_df = val_df.sample(frac=0.1, random_state=43)
    #test_df = test_df.sample(frac=0.1, random_state=44)
    return img_dir, train_df, test_df, val_df

# generating more test data by altering dataset images
def generate_data(img_dir, train_df, test_df, val_df):
    # generator for train data
    train_datagen = ImageDataGenerator(horizontal_flip=True, # mirrors images for more data
                                        rotation_range=20, # rotates images slightly
                                        rescale=1./255.) # rescales values to fractions of 1
    # generator for test data
    test_datagen = ImageDataGenerator(rescale=1./255.)

    TARGET_size = (224, 224)
    BATCH_size = 128

    # generating training data
    train_images = train_datagen.flow_from_dataframe(
        dataframe = train_df,
        directory = img_dir,
        x_col ='image_path',
        y_col ='class_label',
        target_size = TARGET_size,
        color_mode ='rgb',
        class_mode ='categorical',
        batch_size = BATCH_size,
        shuffle = True,
        seed = 33,
        subset ='training'
    )

    # generating test data
    test_images = test_datagen.flow_from_dataframe(
        dataframe = test_df,
        directory = img_dir,
        x_col ='image_path',
        y_col ='class_label',
        target_size = TARGET_size,
        color_mode ='rgb',
        class_mode ='categorical',
        batch_size = BATCH_size,
        shuffle = False
    )

    # generating validation data from training dataset
    val_images = train_datagen.flow_from_dataframe(
        dataframe = val_df,
        directory = img_dir,
        x_col ='image_path',
        y_col ='class_label',
        target_size = TARGET_size,
        color_mode ='rgb',
        class_mode ='categorical',
        batch_size = BATCH_size,
        shuffle = True,
        seed = 97
    )

    return train_images, test_images, val_images

# defining the parameters of pre-trained model to be used
def define_model():
    # clear keras session and releases previously used models from memory, if any
    tf.keras.backend.clear_session()

    # load the model
    model = VGG16(include_top=False, # exclude the pre-trained fully connected feed-forward network
                 pooling='avg', # use average values when pooling inputs 
                 input_shape=(224,224,3))

    # mark remaining loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False

    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(1024, activation='relu')(flat1)
    #class2 = Dense(128, activation='relu')(class1) # extra classification layer commented out due to long runtime
    # add dropout to prevent overfitting
    drop1 = Dropout(0.1)(class1)
    output = Dense(15, activation='softmax')(drop1)

    # define new model
    model = Model(inputs=model.inputs, 
                  outputs=output)
    ###
    # assign learning rates to the weights
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)

    # compile model with chosen parameters
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# training model within set parameters
def train_model(model, train_images, val_images):
    # outlining requirements for stopping the model early if stops improving (by validation loss per default)
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    early_stopping = [early_stopping] # making list for callbacks

    H = model.fit(train_images,
                  validation_data=val_images,
                  batch_size=128, # how many inputs the neural network trains on before updating its weights ###
                  epochs=50,
                  verbose=1,
                  callbacks=early_stopping)
    return H

# plot graph of training history
def plot_graph(model_hist):
    epochs = len(model_hist.history["loss"]) # this is necessary to determine the number of epochs in case early_stopping activates, where a fixed value would fail
    plot = rf.plot_history(model_hist, epochs) # from utils folder, creates and saves plot to out folder
    return None

# save model to set outpath (commented out in main() )
def save_model(model_name):
    outpath = os.path.join("..","out","fashion_model.keras") # defining outpath for results
    tf.keras.models.save_model(
        model_name, outpath, overwrite=False, save_format=None, # set overwrite=True to skip console prompt to overwrite old file
    )
    return None

# making classification report using test data
def make_report(model, test_images, test_df):
    y_test = list(test_df.class_label) # list of ground truth values ###
    predictions = model.predict(test_images, batch_size=128) # creating predictions
    label_names = set(test_df['class_label'])

    # maping best predictions to str labels
    label_mapping = {i: label for i, label in enumerate(label_names)}
    predicted_labels = [label_mapping[prediction] for prediction in predictions.argmax(axis=1)]

    report = classification_report(y_test,
                                   predicted_labels,
                                   target_names=label_names)
    return report

# saving classification report to set outpath
def save_report(report):
    outpath = os.path.join("..","out","classification_report.txt") # defining outpath for report

    # overwrite old report file or create new if none exists
    with open(outpath, 'w') as file:
        file.write(report)
    return None

def main():
    print("Loading images ...")
    img_path, train, test, val = load_image() # assign image path and train/test/val data
    print("Generating data ...")
    train_im, test_im, val_im = generate_data(img_path, train, test, val) # generate train/test/val data

    print("Defining model parameters ...")
    model = define_model() # set new parameters for pre-trained model
    print("Training model ...")
    model_hist = train_model(model, train_im, val_im) # train model, assign hist
    #save_model(model) # save trained model
    #print("Model saved to 'out' folder.")

    plot = plot_graph(model_hist) # make training hist plot and save to out folder
    print("Making classification report ...")
    class_report = make_report(model, test_im, test) # make classification report on test data
    save_report(class_report) # save report
    print("Report and plot saved to 'out' folder.")
    return None

if __name__ == "__main__":
    main()