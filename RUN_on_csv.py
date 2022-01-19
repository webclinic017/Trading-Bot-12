# -*- coding: utf-8 -*-
"""
Created on Sun May  9 12:12:01 2021

@author: Mahdi
"""

import numpy as np
import pandas as pd
# import threading
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from keras.models import load_model

# from time import sleep

# import traceback


def model_predict( model, arr_df, num_ds_row = 96, num_ds_col=60 ) :
    prediction = [0] * num_ds_row
    LU = [0] * num_ds_row
    S = [0] * num_ds_row
    LD = [0] * num_ds_row
    for i in range(num_ds_row, len(arr_df)):
        ctf1 = arr_df[ i-num_ds_row:i , 63: ]
        # print(ctf1)
        model_output = model.predict(np.asarray(ctf1).astype('float32').reshape(1, num_ds_row, num_ds_col))
        # print(model_output)
        LU.append(model_output[0,0])
        S.append(model_output[0,1])
        LD.append(model_output[0,2])
        model_output  = np.argmax(model_output, axis = 1)
        if model_output == 0:
            prediction.append('Short_Lot_Up')
        
        elif model_output == 1:
            prediction.append('Sit')
        
        elif model_output == 2:
            prediction.append('Short_Lot_Down')
        
        # if model_output == 0:
        #     prediction.append('Short_Lot_Up')
        
        # elif model_output == 1:
        #     prediction.append('Short_Lit_Up')
        
        # elif model_output == 2:
        #     prediction.append('Sit')
        
        # elif model_output == 3:
        #     prediction.append('Short_Lit_Down')
        
        # elif model_output == 4:
        #     prediction.append('Short_Lot_Down')
            
    # print(prediction)
    return prediction, LU, S, LD



if __name__ == '__main__':
    num_ds_row = 84
    num_ds_col = 87
    future_type = '5mto6h'
    label_type = '3Up3Dw6SS6SR_8&10'
    features_type = 'wr&roc_nosqr'
    num_label_type = 3 + 3 + 6 + 6
    trade_actions = ['Short_Lot_Up', 'Sit', 'Short_Lot_Down']
    nb_actions = 3
    # trade_actions = ['Short_Lot_Up', 'Short_Lit_Up', 'Sit', 'Short_Lit_Down', 'Short_Lot_Down']
    # nb_actions = 5
    
    #Define the model
    filters = 84
    kernel_size = 6
    model_architecture = '3Conv1Df'+str(filters)+'k'+str(kernel_size)+'psame_MxPl1Do2_Dense128'
    DNN_activation = 'relu'
    last_activation = 'softmax'
    
    batch_size = 2 * num_label_type
    weights_path = None
    
    experiment_name = label_type+'_'+str(batch_size)+'_'+str(num_ds_row)+'_'+str(num_ds_col)+'_'+future_type+'_'+features_type+'_'+model_architecture+'_'+DNN_activation+'_'+last_activation
    
    model_output_dir = 'C:/Users/Mahdi/Documents/spyder/Crypto_Trader/DataDir/classification_output'
    checkpoint_filepath = os.path.join(model_output_dir, experiment_name, '04-0.8470439-0.8692896-0.3962670-0.3153605.h5')
    model = load_model(checkpoint_filepath)
    
    # SymbolsUSDT = ['1INCHUSDT', 'AAVEUSDT', 'ACMUSDT', 'ADAUSDT', 'AIONUSDT', 'AKROUSDT', 'ALGOUSDT', 'ALPHAUSDT', 'ANKRUSDT', 'ANTUSDT', 'ARDRUSDT', 'ARPAUSDT', 'ASRUSDT', 'ATMUSDT', 'ATOMUSDT', 'AUDIOUSDT', 'AUDUSDT', 'AVAUSDT', 'AVAXUSDT', 'AXSUSDT', 'BADGERUSDT', 'BALUSDT', 'BANDUSDT', 'BATUSDT', 'BCHUSDT', 'BEAMUSDT', 'BELUSDT', 'BLZUSDT', 'BNBUSDT', 'BNTUSDT', 'BTCUSDT', 'BTSUSDT', 'BTTUSDT', 'BZRXUSDT', 'CAKEUSDT', 'CELOUSDT', 'CELRUSDT', 'CHRUSDT', 'CHZUSDT', 'CKBUSDT', 'COCOSUSDT', 'COMPUSDT', 'COSUSDT', 'COTIUSDT', 'CRVUSDT', 'CTKUSDT', 'CTSIUSDT', 'CTXCUSDT', 'CVCUSDT', 'DASHUSDT', 'DATAUSDT', 'DCRUSDT', 'DENTUSDT', 'DGBUSDT', 'DIAUSDT', 'DNTUSDT', 'DOCKUSDT', 'DODOUSDT', 'DOGEUSDT', 'DOTUSDT', 'DREPUSDT', 'DUSKUSDT', 'EGLDUSDT', 'ENJUSDT', 'EOSUSDT', 'ETCUSDT', 'ETHUSDT', 'FETUSDT', 'FILUSDT', 'FIOUSDT', 'FIROUSDT', 'FISUSDT', 'FLMUSDT', 'FTMUSDT', 'FTTUSDT', 'FUNUSDT', 'GRTUSDT', 'GTOUSDT', 'GXSUSDT', 'HARDUSDT', 'HBARUSDT', 'HIVEUSDT', 'HNTUSDT', 'HOTUSDT', 'ICXUSDT', 'INJUSDT', 'IOSTUSDT', 'IOTAUSDT', 'IOTXUSDT', 'IRISUSDT', 'JSTUSDT', 'JUVUSDT', 'KAVAUSDT', 'KEYUSDT', 'KMDUSDT', 'KNCUSDT', 'KSMUSDT', 'LINKUSDT', 'LITUSDT', 'LRCUSDT', 'LSKUSDT', 'LTCUSDT', 'LTOUSDT', 'LUNAUSDT', 'MANAUSDT', 'MATICUSDT', 'MBLUSDT', 'MDTUSDT', 'MFTUSDT', 'MITHUSDT', 'MKRUSDT', 'MTLUSDT', 'NANOUSDT', 'NBSUSDT', 'NEARUSDT', 'NEOUSDT', 'NKNUSDT', 'NMRUSDT', 'NPXSUSDT', 'NULSUSDT', 'OCEANUSDT', 'OGNUSDT', 'OMGUSDT', 'ONEUSDT', 'ONGUSDT', 'ONTUSDT', 'ORNUSDT', 'OXTUSDT', 'PAXGUSDT', 'PAXUSDT', 'PERLUSDT', 'PNTUSDT', 'PSGUSDT', 'QTUMUSDT', 'REEFUSDT', 'RENUSDT', 'REPUSDT', 'RIFUSDT', 'RLCUSDT', 'ROSEUSDT', 'RSRUSDT', 'RUNEUSDT', 'RVNUSDT', 'SANDUSDT', 'SFPUSDT', 'SKLUSDT', 'SNXUSDT', 'SOLUSDT', 'SRMUSDT', 'STMXUSDT', 'STORJUSDT', 'STPTUSDT', 'STRAXUSDT', 'STXUSDT', 'SUNUSDT', 'SUSDUSDT', 'SUSHIUSDT', 'SXPUSDT', 'TCTUSDT', 'TFUELUSDT', 'THETAUSDT', 'TOMOUSDT', 'TRBUSDT', 'TROYUSDT', 'TRUUSDT', 'TRXUSDT', 'TWTUSDT', 'UMAUSDT', 'UNFIUSDT', 'UNIUSDT', 'UTKUSDT', 'VETUSDT', 'VITEUSDT', 'VTHOUSDT', 'WANUSDT', 'WAVESUSDT', 'WINGUSDT', 'WINUSDT', 'WNXMUSDT', 'WRXUSDT', 'WTCUSDT', 'XEMUSDT', 'XLMUSDT', 'XMRUSDT', 'XRPUSDT', 'XTZUSDT', 'XVSUSDT', 'YFIIUSDT', 'YFIUSDT', 'ZECUSDT', 'ZENUSDT', 'ZILUSDT', 'ZRXUSDT']
    # SymbolsUSDT = ['ALPHAUSDT', 'MATICUSDT', 'LINKUSDT', 'XRPUSDT']
    SymbolsUSDT = ['BADGERUSDT']
    print(len(SymbolsUSDT))
    
    for symbol in SymbolsUSDT:
        print('Symbol: ' + symbol)
        c_df_h = pd.read_csv(symbol + '.csv', index_col=0)
        # c_df_h.drop(['model_prediction'], inplace=True, axis=1)
        model_prediction, LU, S, LD = model_predict( model, c_df_h.to_numpy() , num_ds_row, num_ds_col)
        c_df_h.insert(5, 'model_prediction', model_prediction)
        c_df_h.insert(6, 'LU', LU)
        c_df_h.insert(7, 'S', S)
        c_df_h.insert(8, 'LD', LD)
        # export DataFrame to csv
        c_df_h.to_csv(symbol + '.csv')
        