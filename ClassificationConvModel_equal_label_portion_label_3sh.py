# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:45:19 2021

@author: Mahdi
"""

import numpy as np
# import threading
import os
import random

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Conv1D, MaxPool1D, Dropout, Flatten, Dense, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam#, SGD, Adamax, Nadam, Adagrad, Adadelta
from keras.losses import categorical_crossentropy
from keras.initializers import random_normal
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from keras.utils import Sequence

from sklearn.model_selection import train_test_split

from PlotHistory import PlotHistory


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, ShLotUp1_IDs, ShLotUp2_IDs, ShLotUp3_IDs, #ShLotUp4_IDs,
                 ShLotDw1_IDs, ShLotDw2_IDs, ShLotDw3_IDs, #ShLotDw4_IDs, 
                 SitS1_IDs, SitS2_IDs, SitS3_IDs, SitS4_IDs, SitS5_IDs, SitS6_IDs,
                 SitR1_IDs, SitR2_IDs, SitR3_IDs, SitR4_IDs, SitR5_IDs, SitR6_IDs,
                 dataset_dir, batch_size=72, dim=(96,60), n_classes=3, num_label_type=18, shuffle=True):
        'Initialization'
        self.ShLotUp1_IDs = ShLotUp1_IDs
        self.ShLotUp2_IDs = ShLotUp2_IDs
        self.ShLotUp3_IDs = ShLotUp3_IDs
        # self.ShLotUp4_IDs = ShLotUp4_IDs
        
        self.ShLotDw1_IDs = ShLotDw1_IDs
        self.ShLotDw2_IDs = ShLotDw2_IDs
        self.ShLotDw3_IDs = ShLotDw3_IDs
        # self.ShLotDw4_IDs = ShLotDw4_IDs
        
        self.SitS1_IDs = SitS1_IDs
        self.SitS2_IDs = SitS2_IDs
        self.SitS3_IDs = SitS3_IDs
        self.SitS4_IDs = SitS4_IDs
        self.SitS5_IDs = SitS5_IDs
        self.SitS6_IDs = SitS6_IDs
        
        self.SitR1_IDs = SitR1_IDs
        self.SitR2_IDs = SitR2_IDs
        self.SitR3_IDs = SitR3_IDs
        self.SitR4_IDs = SitR4_IDs
        self.SitR5_IDs = SitR5_IDs
        self.SitR6_IDs = SitR6_IDs
        
        self.len_ShLotUp1 = len(self.ShLotUp1_IDs)
        self.len_ShLotUp2 = len(self.ShLotUp2_IDs)
        self.len_ShLotUp3 = len(self.ShLotUp3_IDs)
        # self.len_ShLotUp4 = len(self.ShLotUp4_IDs)
        
        self.len_ShLotDw1 = len(self.ShLotDw1_IDs)
        self.len_ShLotDw2 = len(self.ShLotDw2_IDs)
        self.len_ShLotDw3 = len(self.ShLotDw3_IDs)
        # self.len_ShLotDw4 = len(self.ShLotDw4_IDs)
        
        self.len_SitS1 = len(self.SitS1_IDs)
        self.len_SitS2 = len(self.SitS2_IDs)
        self.len_SitS3 = len(self.SitS3_IDs)
        self.len_SitS4 = len(self.SitS4_IDs)
        self.len_SitS5 = len(self.SitS5_IDs)
        self.len_SitS6 = len(self.SitS6_IDs)
        
        self.len_SitR1 = len(self.SitR1_IDs)
        self.len_SitR2 = len(self.SitR2_IDs)
        self.len_SitR3 = len(self.SitR3_IDs)
        self.len_SitR4 = len(self.SitR4_IDs)
        self.len_SitR5 = len(self.SitR5_IDs)
        self.len_SitR6 = len(self.SitR6_IDs)
        
        self.n = max(self.len_ShLotUp1, self.len_ShLotDw1, self.len_SitS1, self.len_SitR1)
        self.shuffle = shuffle
        self.dim = dim
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.num_label_type = num_label_type
        self.inner_batch_size = batch_size//self.num_label_type
        self.dataset_dir = dataset_dir
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n/self.inner_batch_size) - 1)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes_temp = self.indexes[index*self.inner_batch_size:(index+1)*self.inner_batch_size]

        # Find list of IDs
        list_IDs_temp = [self.ShLotUp1_IDs[k % self.len_ShLotUp1] for k in indexes_temp]
        list_IDs_temp += [self.ShLotUp2_IDs[k % self.len_ShLotUp2] for k in indexes_temp]
        list_IDs_temp += [self.ShLotUp3_IDs[k % self.len_ShLotUp3] for k in indexes_temp]
        # list_IDs_temp += [self.ShLotUp4_IDs[k % self.len_ShLotUp4] for k in indexes_temp]
        
        list_IDs_temp += [self.ShLotDw1_IDs[k % self.len_ShLotDw1] for k in indexes_temp]
        list_IDs_temp += [self.ShLotDw2_IDs[k % self.len_ShLotDw2] for k in indexes_temp]
        list_IDs_temp += [self.ShLotDw3_IDs[k % self.len_ShLotDw3] for k in indexes_temp]
        # list_IDs_temp += [self.ShLotDw4_IDs[k % self.len_ShLotDw4] for k in indexes_temp]
        
        list_IDs_temp += [self.SitS1_IDs[k % self.len_SitS1] for k in indexes_temp]
        list_IDs_temp += [self.SitS2_IDs[k % self.len_SitS2] for k in indexes_temp]
        list_IDs_temp += [self.SitS3_IDs[k % self.len_SitS3] for k in indexes_temp]
        list_IDs_temp += [self.SitS4_IDs[k % self.len_SitS4] for k in indexes_temp]
        list_IDs_temp += [self.SitS5_IDs[k % self.len_SitS5] for k in indexes_temp]
        list_IDs_temp += [self.SitS6_IDs[k % self.len_SitS6] for k in indexes_temp]
        
        list_IDs_temp += [self.SitR1_IDs[k % self.len_SitR1] for k in indexes_temp]
        list_IDs_temp += [self.SitR2_IDs[k % self.len_SitR2] for k in indexes_temp]
        list_IDs_temp += [self.SitR3_IDs[k % self.len_SitR3] for k in indexes_temp]
        list_IDs_temp += [self.SitR4_IDs[k % self.len_SitR4] for k in indexes_temp]
        list_IDs_temp += [self.SitR5_IDs[k % self.len_SitR5] for k in indexes_temp]
        list_IDs_temp += [self.SitR6_IDs[k % self.len_SitR6] for k in indexes_temp]
        
        random.shuffle(list_IDs_temp)

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.n) # max
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            batch_dir = os.path.join(self.dataset_dir,'{0}.npz'.format(ID))
            dic = np.load(batch_dir)
            X[i,] = dic['dataset']

            # Store class
            y[i] = dic['label']

        return X, y


if __name__ == '__main__':
    num_ds_row = 84
    num_ds_col = 87
    future_type = '5mto6h'
    label_type = '3Up3Dw6SS6SR_8&10'
    features_type = 'wr&roc_nosqr'
    num_label_type = 3 + 3 + 6 + 6
    trade_actions = ['Short_Lot_Up', 'Sit', 'Short_Lot_Down']
    nb_actions = 3
    
    #Define the model
    filters = 84
    kernel_size = 6
    model_architecture = '3Conv1Df'+str(filters)+'k'+str(kernel_size)+'psame_MxPl1Do2_Dense128'
    DNN_activation = 'relu'
    last_activation = 'softmax'
    
    batch_size = 2 * num_label_type
    weights_path = None
    
    experiment_name = label_type+'_'+str(batch_size)+'_'+str(num_ds_row)+'_'+str(num_ds_col)+'_'+future_type+'_'+features_type+'_'+model_architecture+'_'+DNN_activation+'_'+last_activation
    
    
    
    # dic_tr_val = np.load('Classification_tr_val_24_28_Conv.npz')
    # print(dic_tr_val['dataset'].shape )
    # print( dic_tr_val['label'].shape )
    dataset_dir = os.path.join('C:/Users/Mahdi/Documents/spyder/Crypto_Trader/DataDir/classification_mono_dataset',str(num_ds_row)+'_'+str(num_ds_col)+'_'+future_type+'_'+features_type)
    dic_IDs = np.load('monoIDs_'+label_type+'_'+str(num_ds_row)+'_'+str(num_ds_col)+'_'+future_type+'_'+features_type+'.npz')
    
    ShLotUp1_train_IDs, ShLotUp1_eval_IDs = train_test_split(dic_IDs['ShLotUp1_IDs'], test_size=0.2)
    ShLotUp2_train_IDs, ShLotUp2_eval_IDs = train_test_split(dic_IDs['ShLotUp2_IDs'], test_size=0.2)
    ShLotUp3_train_IDs, ShLotUp3_eval_IDs = train_test_split(dic_IDs['ShLotUp3_IDs'], test_size=0.2)
    # ShLotUp4_train_IDs, ShLotUp4_eval_IDs = train_test_split(dic_IDs['ShLotUp4_IDs'], test_size=0.2)
    
    ShLotDw1_train_IDs, ShLotDw1_eval_IDs = train_test_split(dic_IDs['ShLotDw1_IDs'], test_size=0.2)
    ShLotDw2_train_IDs, ShLotDw2_eval_IDs = train_test_split(dic_IDs['ShLotDw2_IDs'], test_size=0.2)
    ShLotDw3_train_IDs, ShLotDw3_eval_IDs = train_test_split(dic_IDs['ShLotDw3_IDs'], test_size=0.2)
    # ShLotDw4_train_IDs, ShLotDw4_eval_IDs = train_test_split(dic_IDs['ShLotDw4_IDs'], test_size=0.2)
    
    SitS1_train_IDs, SitS1_eval_IDs = train_test_split(dic_IDs['SitS1_IDs'], test_size=0.2)
    SitS2_train_IDs, SitS2_eval_IDs = train_test_split(dic_IDs['SitS2_IDs'], test_size=0.2)
    SitS3_train_IDs, SitS3_eval_IDs = train_test_split(dic_IDs['SitS3_IDs'], test_size=0.2)
    SitS4_train_IDs, SitS4_eval_IDs = train_test_split(dic_IDs['SitS4_IDs'], test_size=0.2)
    SitS5_train_IDs, SitS5_eval_IDs = train_test_split(dic_IDs['SitS5_IDs'], test_size=0.2)
    SitS6_train_IDs, SitS6_eval_IDs = train_test_split(dic_IDs['SitS6_IDs'], test_size=0.2)
    
    SitR1_train_IDs, SitR1_eval_IDs = train_test_split(dic_IDs['SitR1_IDs'], test_size=0.2)
    SitR2_train_IDs, SitR2_eval_IDs = train_test_split(dic_IDs['SitR2_IDs'], test_size=0.2)
    SitR3_train_IDs, SitR3_eval_IDs = train_test_split(dic_IDs['SitR3_IDs'], test_size=0.2)
    SitR4_train_IDs, SitR4_eval_IDs = train_test_split(dic_IDs['SitR4_IDs'], test_size=0.2)
    SitR5_train_IDs, SitR5_eval_IDs = train_test_split(dic_IDs['SitR5_IDs'], test_size=0.2)
    SitR6_train_IDs, SitR6_eval_IDs = train_test_split(dic_IDs['SitR6_IDs'], test_size=0.2)
    
    train_generator = DataGenerator( ShLotUp1_train_IDs, ShLotUp2_train_IDs, ShLotUp3_train_IDs, #ShLotUp4_train_IDs,
                                    ShLotDw1_train_IDs, ShLotDw2_train_IDs, ShLotDw3_train_IDs, #ShLotDw4_train_IDs, 
                                    SitS1_train_IDs, SitS2_train_IDs, SitS3_train_IDs, SitS4_train_IDs, SitS5_train_IDs, SitS6_train_IDs,
                                    SitR1_train_IDs, SitR2_train_IDs, SitR3_train_IDs, SitR4_train_IDs, SitR5_train_IDs, SitR6_train_IDs,
                                    dataset_dir, batch_size=batch_size, dim=(num_ds_row, num_ds_col), n_classes=nb_actions, num_label_type=num_label_type, shuffle=True)
    eval_generator = DataGenerator( ShLotUp1_eval_IDs, ShLotUp2_eval_IDs, ShLotUp3_eval_IDs, #ShLotUp4_eval_IDs,
                                   ShLotDw1_eval_IDs, ShLotDw2_eval_IDs, ShLotDw3_eval_IDs, #ShLotDw4_eval_IDs,
                                   SitS1_eval_IDs, SitS2_eval_IDs, SitS3_eval_IDs, SitS4_eval_IDs, SitS5_eval_IDs, SitS6_eval_IDs,
                                   SitR1_eval_IDs, SitR2_eval_IDs, SitR3_eval_IDs, SitR4_eval_IDs, SitR5_eval_IDs, SitR6_eval_IDs,
                                   dataset_dir, batch_size=batch_size, dim=(num_ds_row, num_ds_col), n_classes=nb_actions, num_label_type=num_label_type, shuffle=True)
    
    
    ds_input = Input(shape=(num_ds_row,num_ds_col), name='ds_input')
    
    ds_stack = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', activation=DNN_activation, name='ds_conv1d0')(ds_input)
    ds_stack = MaxPool1D(pool_size=2, padding="valid")(ds_stack)
    ds_stack = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation=DNN_activation, name='ds_conv1d1')(ds_stack)
    ds_stack = MaxPool1D(pool_size=2, padding="valid")(ds_stack)
    ds_stack = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation=DNN_activation, name='ds_conv1d2')(ds_stack)
    ds_stack = MaxPool1D(pool_size=2, padding="valid")(ds_stack)
    # ds_input = Input(shape=(num_ds_row, num_ds_col, 1), name='ds_input')
    # ds_stack = Conv2D(filters=32, kernel_size=(2,2), strides=(1,2), padding='valid', activation=DNN_activation, name='ds_conv2d0')(ds_input)
    # # ( num_ds_col/2 , filters )
    ds_stack = Flatten()(ds_stack)
    ds_stack = Dense(128, activation=DNN_activation, name='ds_dense0')(ds_stack)
    
    # ds_stack = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', name='ds_conv1d0')(ds_input)
    # ds_stack = LeakyReLU(alpha=0.1)(ds_stack)
    # ds_stack = MaxPool1D(pool_size=2, padding="valid")(ds_stack)
    # ds_stack = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', name='ds_conv1d1')(ds_stack)
    # ds_stack = LeakyReLU(alpha=0.1)(ds_stack)
    # ds_stack = MaxPool1D(pool_size=2, padding="valid")(ds_stack)
    # ds_stack = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', name='ds_conv1d2')(ds_stack)
    # ds_stack = LeakyReLU(alpha=0.1)(ds_stack)
    # ds_stack = MaxPool1D(pool_size=2, padding="valid")(ds_stack)
    # # ds_input = Input(shape=(num_ds_row, num_ds_col, 1), name='ds_input')
    # # ds_stack = Conv2D(filters=32, kernel_size=(2,2), strides=(1,2), padding='valid', activation=DNN_activation, name='ds_conv2d0')(ds_input)
    # # # ( num_ds_col/2 , filters )
    # ds_stack = Flatten()(ds_stack)
    # ds_stack = Dense(128, name='ds_dense0')(ds_stack)
    # ds_stack = LeakyReLU(alpha=0.1)(ds_stack)
    
    output = Dense(nb_actions, activation=last_activation, name='output')(ds_stack)
    
    opt = Adam()
    model = Model(inputs=ds_input, outputs=output)
    
    model.compile(optimizer=opt, loss= categorical_crossentropy, metrics = ['accuracy'])
    model.summary()
    
    
    # If we are using pretrained weights for the conv layers, load them.
    if (weights_path is not None and len(weights_path) > 0):
        print('Loading weights')
        print('Current working dir is {0}'.format(os.getcwd()))
        model.load_weights(weights_path, by_name=True)
    else:
        print('Not loading weights')
    
    
    plateau_callback = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=4, min_lr=0.0001, verbose=1)
    
    model_output_dir = 'C:/Users/Mahdi/Documents/spyder/Crypto_Trader/DataDir/classification_output'
    checkpoint_filepath = os.path.join(model_output_dir, experiment_name, '{0}-{1}-{2}-{3}-{4}.h5'.format('{epoch:02d}', '{val_accuracy:.7f}', '{accuracy:.7f}', '{val_loss:.7f}', '{loss:.7f}'))
    if not os.path.exists(os.path.dirname(checkpoint_filepath)):
        try:
            os.makedirs(os.path.dirname(checkpoint_filepath))
        # except OSError as exc:  # Guard against race condition
        #     if exc.errno != errno.EEXIST:
        #         raise
        except OSError:  # Guard against race condition
            raise
        
    checkpoint_callback = ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', save_best_only=True, verbose=1)
    
    early_stopping_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=3, verbose=1)
    
    callbacks = [plateau_callback, checkpoint_callback, early_stopping_callback]
    
    history = model.fit(train_generator, steps_per_epoch=len(train_generator),
                        epochs=8, verbose=2, callbacks=callbacks,
                        validation_data=eval_generator, validation_steps=len(eval_generator),
                        use_multiprocessing=True, workers=9)
    
    # history = model.fit(dic_tr_val['dataset'], dic_tr_val['label'],
    #                     batch_size=batch_size, epochs=100, verbose=2, callbacks=callbacks,
    #                     shuffle=True, validation_split = 0.2)
    PlotHistory(history)