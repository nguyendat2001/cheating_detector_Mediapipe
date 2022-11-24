# training model
# python train.py --train ./data/landmarks/train --test ./data/landmarks/test --output ./figures --epochs 500 --batch 64 --patience 20

import argparse
import cv2
import numpy as np
import os
import time
import pandas as pd
import mediapipe as mp
import threading
import tensorflow as tf

from tensorflow import keras
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *



# point_landmark_toget = [0, 11 ,12 ,13, 14,15 ,16, 19, 20, 23, 24]

def getmodel(X):
    model  = Sequential()
    model.add(LSTM(units = 64, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 32, return_sequences = True))
    model.add(Dropout(0.2))
    # model.add(LSTM(units = 16, return_sequences = True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units = 50))
    # model.add(Dropout(0.2))
    model.add(Dense(units = 64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units = 1, activation="sigmoid"))
    model.summary()
    model.compile(optimizer="adam", metrics = ['accuracy'], loss = "binary_crossentropy")
    return model

def plot_confusion_matrix(plot_confusion_matrix_path,cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """Plots the confusion matrix."""
        if normalize:
          cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
          print("Normalized confusion matrix")
        else:
          print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=55)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
          plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(plot_confusion_matrix_path)
        plt.close()
        

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
#     parser.add_argument('', type=str, default='yolo7.pt', help='initial weights path')
#     parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--train','-d', type=str, default='./data', help='path')
    parser.add_argument('--test','-t', type=str, default='./test', help='training or testing data')
    parser.add_argument('--output','-o', type=str, default='./figures', help='output of training result')
    parser.add_argument('--epochs','-e', type=int, default=500)
    parser.add_argument('--batch','-b', type=int, default=64)
    parser.add_argument('--patience','-p', type=int, default=15)
    opt = parser.parse_args()
    
    
#     cheat = pd.read_csv(os.path.join(path_data,'/landmarks/train','cheat.csv'))
#     non_cheat = pd.read_csv(os.path.join(path_data,'/landmarks/train',"non_cheat.csv")
    cheat = pd.read_csv(os.path.join(opt.train,'cheat.csv'))
    non_cheat = pd.read_csv(os.path.join(opt.train,"non_cheat.csv"))

    X = []
    y = []
    no_of_timesteps = 10

    dataset = cheat.iloc[:,1:].values
    n_sample = len(dataset)
    for i in range(no_of_timesteps, n_sample):
        X.append(dataset[i-no_of_timesteps:i,:])
        y.append(1)

    dataset = non_cheat.iloc[:,1:].values
    n_sample = len(dataset)
    for i in range(no_of_timesteps, n_sample):
        X.append(dataset[i-no_of_timesteps:i,:])
        y.append(0)


    # test
#     cheat = pd.read_csv(os.path.join(path_data,'/landmarks/test','cheat.csv'))
#     non_cheat = pd.read_csv(os.path.join(path_data,'/landmarks/test',"non_cheat.csv")
    cheat = pd.read_csv(os.path.join(opt.test,'cheat.csv'))
    non_cheat = pd.read_csv(os.path.join(opt.test,"non_cheat.csv"))
                            
    X_test = []
    y_test = []
    no_of_timesteps = 10

    dataset = cheat.iloc[:,1:].values
    n_sample = len(dataset)
    for i in range(no_of_timesteps, n_sample):
        X_test.append(dataset[i-no_of_timesteps:i,:])
        y_test.append(1)

    dataset = non_cheat.iloc[:,1:].values
    n_sample = len(dataset)
    for i in range(no_of_timesteps, n_sample):
        X_test.append(dataset[i-no_of_timesteps:i,:])
        y_test.append(0)
                            
                        
    X, y = np.array(X), np.array(y)
    print(X.shape, y.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    X_test, y_test = np.array(X_test), np.array(y_test)
    
    figure_data_path = opt.output
    model = getmodel(X_train)

    checkpoint_path = os.path.join(figure_data_path,"weights.best.hdf5")
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                              patience = opt.patience)

    history = model.fit(X_train,y_train ,
              batch_size=opt.batch,
              epochs = opt.epochs,
              validation_data=(X_val, y_val),
              callbacks=[checkpoint,earlystopping])


    model.save(os.path.join(figure_data_path,"model.h5"))
      
    loss_test,accuracy_test = model.evaluate(X_test, y_test)
    print('LOSS TEST: ', loss_test)
    print("ACCURACY TEST: ", accuracy_test)


    loss_train, accuracy_train = model.evaluate(X_train, y_train)
    print('LOSS TRAIN: ', loss_train)
    print("ACCURACY TRAIN: ", accuracy_train)


    data_eval = 'LOSS TEST: '+ str(loss_test) + ' \n ACCURACY TEST: '+ str(accuracy_test) + '\n' +'LOSS TRAIN: '+ str(loss_train) +'\n ACCURACY TRAIN: '+ str(accuracy_train) +"\n time : "+str((time_end - time_start)/60)

    hist_df = pd.DataFrame(history.history)
    name_history = 'history.csv'
    path_history = os.path.join(figure_data_path , name_history)
    with open(path_history, mode='w') as f:
        hist_df.to_csv(f)

    # write EVAUATION into txt
    Name_f = 'EVAUATION.txt'
    # f = open('./figure/CNN/train_1st/EVAUATION_1st.txt','w+')

    path_s= os.path.join(figure_data_path , Name_f)

    with open(path_s, mode='w') as f:
        f.writelines(data_eval)
    f.close()
        
    
    