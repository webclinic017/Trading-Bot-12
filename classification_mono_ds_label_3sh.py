# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:37:54 2021

@author: Mahdi
"""

import numpy as np
import os


import pandas as pd

import concurrent.futures
import traceback

    
def get_df_batch(c_df_h, period_h, init):
    df_h = c_df_h
    max_h = len(df_h)
    n_h = init
    # An Infinite loop to generate squares
    while period_h + n_h + 13 <= max_h:
        df_h_b = df_h.iloc[n_h : period_h + n_h , : ]
        n_h += 1
        yield df_h_b

def process_ds(args):
    try:
        symbol, init, dataset_dir, num_ds_row, num_ds_col = args
        
        ShLotUp = []
        SitR = []
        SitS = []
        ShLotDw = []
        
        
        ds_count = 0
        s=True
        dic_tr_val = {}
        dic_tr_val['dataset'] = 0
        dic_tr_val['label'] = 0
        c_df_h = pd.read_csv(symbol + '.csv', index_col=0)
        # Using for to get all generator element
        for df_h in get_df_batch(c_df_h, num_ds_row, init):
            h = df_h.iloc[ : , 63: ]
            if s:
                print(h)
                s=False
            dic_tr_val['dataset'] = h.to_numpy()
            
            label = df_h.iloc[-1]['Labels']
            
            ID = symbol+'_'+str(ds_count)
                
            if label == 'Short_Lot_Up':
                dic_tr_val['label'] = np.array([1, 0, 0], copy=False)
                ShLotUp.append(ID)
            
            elif label == 'Sit_for_Resistance':
                dic_tr_val['label'] = np.array([0, 1, 0], copy=False)
                SitR.append(ID)
            
            elif label == 'Sit_for_support':
                dic_tr_val['label'] = np.array([0, 1, 0], copy=False)
                SitS.append(ID)
            
            elif label == 'Short_Lot_Down':
                dic_tr_val['label'] = np.array([0, 0, 1], copy=False)
                ShLotDw.append(ID)
            
            batch_dir = os.path.join(dataset_dir,'{0}.npz'.format(ID))
            np.savez_compressed(batch_dir, **dic_tr_val)
            
            # IDs.append(ID)
            
            ds_count += 1
            
        return symbol, [ShLotUp, SitR, SitS, ShLotDw]
        # return symbol, [LGUP, SHUP, LGDW, SHDW, Sit]
    except:
        # printing stack trace
        traceback.print_exc()
        return symbol, 'ZZZ'



if __name__ == '__main__':
    
    dataset_dir = os.path.join('C:/Users/Mahdi/Documents/spyder/Crypto_Trader/DataDir', 'classification_mono_dataset')
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    
    num_ds_row = 84
    num_ds_col = 87
    future_type = '5mto6h'
    label_type = '3Up3Dw6SS6SR_8&10'
    features_type = 'wr&roc_nosqr'
    
    dataset_dir = os.path.join('C:/Users/Mahdi/Documents/spyder/Crypto_Trader/DataDir/classification_mono_dataset',str(num_ds_row)+'_'+str(num_ds_col)+'_'+future_type+'_'+features_type)
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    
    # # All
    # SymbolsUSDT = ['1INCHUSDT', 'AAVEUSDT', 'ACMUSDT', 'ADAUSDT', 'AIONUSDT', 'AKROUSDT', 'ALGOUSDT', 'ALPHAUSDT', 'ANKRUSDT', 'ANTUSDT', 'ARDRUSDT', 'ARPAUSDT', 'ASRUSDT', 'ATMUSDT', 'ATOMUSDT', 'AUDIOUSDT', 'AUDUSDT', 'AVAUSDT', 'AVAXUSDT', 'AXSUSDT', 'BADGERUSDT', 'BALUSDT', 'BANDUSDT', 'BATUSDT', 'BCHUSDT', 'BEAMUSDT', 'BELUSDT', 'BLZUSDT', 'BNBUSDT', 'BNTUSDT', 'BTCUSDT', 'BTSUSDT', 'BTTUSDT', 'BZRXUSDT', 'CAKEUSDT', 'CELOUSDT', 'CELRUSDT', 'CHRUSDT', 'CHZUSDT', 'CKBUSDT', 'COCOSUSDT', 'COMPUSDT', 'COSUSDT', 'COTIUSDT', 'CRVUSDT', 'CTKUSDT', 'CTSIUSDT', 'CTXCUSDT', 'CVCUSDT', 'DASHUSDT', 'DATAUSDT', 'DCRUSDT', 'DENTUSDT', 'DGBUSDT', 'DIAUSDT', 'DNTUSDT', 'DOCKUSDT', 'DODOUSDT', 'DOGEUSDT', 'DOTUSDT', 'DREPUSDT', 'DUSKUSDT', 'EGLDUSDT', 'ENJUSDT', 'EOSUSDT', 'ETCUSDT', 'ETHUSDT', 'FETUSDT', 'FILUSDT', 'FIOUSDT', 'FIROUSDT', 'FISUSDT', 'FLMUSDT', 'FTMUSDT', 'FTTUSDT', 'FUNUSDT', 'GRTUSDT', 'GTOUSDT', 'GXSUSDT', 'HARDUSDT', 'HBARUSDT', 'HIVEUSDT', 'HNTUSDT', 'HOTUSDT', 'ICXUSDT', 'INJUSDT', 'IOSTUSDT', 'IOTAUSDT', 'IOTXUSDT', 'IRISUSDT', 'JSTUSDT', 'JUVUSDT', 'KAVAUSDT', 'KEYUSDT', 'KMDUSDT', 'KNCUSDT', 'KSMUSDT', 'LINKUSDT', 'LITUSDT', 'LRCUSDT', 'LSKUSDT', 'LTCUSDT', 'LTOUSDT', 'LUNAUSDT', 'MANAUSDT', 'MATICUSDT', 'MBLUSDT', 'MDTUSDT', 'MFTUSDT', 'MITHUSDT', 'MKRUSDT', 'MTLUSDT', 'NANOUSDT', 'NBSUSDT', 'NEARUSDT', 'NEOUSDT', 'NKNUSDT', 'NMRUSDT', 'NPXSUSDT', 'NULSUSDT', 'OCEANUSDT', 'OGNUSDT', 'OMGUSDT', 'ONEUSDT', 'ONGUSDT', 'ONTUSDT', 'ORNUSDT', 'OXTUSDT', 'PAXGUSDT', 'PAXUSDT', 'PERLUSDT', 'PNTUSDT', 'PSGUSDT', 'QTUMUSDT', 'REEFUSDT', 'RENUSDT', 'REPUSDT', 'RIFUSDT', 'RLCUSDT', 'ROSEUSDT', 'RSRUSDT', 'RUNEUSDT', 'RVNUSDT', 'SANDUSDT', 'SFPUSDT', 'SKLUSDT', 'SNXUSDT', 'SOLUSDT', 'SRMUSDT', 'STMXUSDT', 'STORJUSDT', 'STPTUSDT', 'STRAXUSDT', 'STXUSDT', 'SUNUSDT', 'SUSDUSDT', 'SUSHIUSDT', 'SXPUSDT', 'TCTUSDT', 'TFUELUSDT', 'THETAUSDT', 'TOMOUSDT', 'TRBUSDT', 'TROYUSDT', 'TRUUSDT', 'TRXUSDT', 'TWTUSDT', 'UMAUSDT', 'UNFIUSDT', 'UNIUSDT', 'UTKUSDT', 'VETUSDT', 'VITEUSDT', 'VTHOUSDT', 'WANUSDT', 'WAVESUSDT', 'WINGUSDT', 'WINUSDT', 'WNXMUSDT', 'WRXUSDT', 'WTCUSDT', 'XEMUSDT', 'XLMUSDT', 'XMRUSDT', 'XRPUSDT', 'XTZUSDT', 'XVSUSDT', 'YFIIUSDT', 'YFIUSDT', 'ZECUSDT', 'ZENUSDT', 'ZILUSDT', 'ZRXUSDT']
    # # 113
    SymbolsUSDT = ['FISUSDT', 'BADGERUSDT', 'DODOUSDT', 'CAKEUSDT', 'SFPUSDT', 'LITUSDT', 'FIROUSDT', 'TWTUSDT', 'CKBUSDT', 'TRUUSDT', 'RIFUSDT', 'CELOUSDT', 'ASRUSDT', 'ATMUSDT', 'REEFUSDT', '1INCHUSDT', 'PSGUSDT', 'GRTUSDT', 'SUSDUSDT', 'SKLUSDT', 'AVAUSDT', 'XEMUSDT', 'ROSEUSDT', 'UNFIUSDT', 'STRAXUSDT', 'DNTUSDT', 'HARDUSDT', 'AXSUSDT', 'AKROUSDT', 'CTKUSDT', 'AUDIOUSDT', 'INJUSDT', 'FILUSDT', 'AAVEUSDT', 'NEARUSDT', 'ALPHAUSDT', 'XVSUSDT', 'ORNUSDT', 'AVAXUSDT', 'SUNUSDT', 'NBSUSDT', 'OXTUSDT', 'UNIUSDT', 'WINGUSDT', 'BELUSDT', 'UMAUSDT', 'DIAUSDT', 'RUNEUSDT', 'EGLDUSDT', 'SUSHIUSDT', 'YFIIUSDT', 'BZRXUSDT', 'TRBUSDT', 'PAXGUSDT', 'LUNAUSDT', 'NMRUSDT', 'DOTUSDT', 'CRVUSDT', 'SANDUSDT', 'SRMUSDT', 'BALUSDT', 'YFIUSDT', 'MANAUSDT', 'STORJUSDT', 'MKRUSDT', 'SXPUSDT', 'VTHOUSDT', 'SNXUSDT', 'ZENUSDT', 'COMPUSDT', 'KNCUSDT', 'LRCUSDT', 'DATAUSDT', 'COTIUSDT', 'LTOUSDT', 'LSKUSDT', 'WRXUSDT', 'DREPUSDT', 'OGNUSDT', 'VITEUSDT', 'RLCUSDT', 'ARPAUSDT', 'XTZUSDT', 'BANDUSDT', 'CVCUSDT', 'DENTUSDT', 'DOCKUSDT', 'FUNUSDT', 'MTLUSDT', 'NPXSUSDT', 'WINUSDT', 'ANKRUSDT', 'DOGEUSDT', 'GTOUSDT', 'TFUELUSDT', 'MATICUSDT', 'MITHUSDT', 'ENJUSDT', 'THETAUSDT', 'NANOUSDT', 'CELRUSDT', 'IOSTUSDT', 'ZECUSDT', 'BATUSDT', 'FETUSDT', 'ZRXUSDT', 'ZILUSDT', 'HOTUSDT', 'WAVESUSDT', 'LINKUSDT', 'XRPUSDT', 'ADAUSDT', 'BNBUSDT']
    # # test
    # SymbolsUSDT = ['ALPHAUSDT']
    print(len(SymbolsUSDT))
    init = {}
    for symbol in SymbolsUSDT:
        init[symbol] = 1
    args = ((symbol, init[symbol], dataset_dir, num_ds_row, num_ds_col) for symbol in SymbolsUSDT)
    done_symbols = []
    ShLotUp = []
    SitR = []
    SitS = []
    ShLotDw = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        results = executor.map(process_ds, args)
    for symbol, IDs in results:
        if IDs != 'ZZZ':
            done_symbols.append(symbol)
            ShLotUp += IDs[0]
            SitR += IDs[1]
            SitS += IDs[2]
            ShLotDw += IDs[3]
        else:
            print('sth wrong with: '+symbol)
    
    splited_ShLotUp = [ShLotUp[ i : i+len(ShLotUp)//3 ] for i in range(0, len(ShLotUp), len(ShLotUp)//3)]
    splited_ShLotDw = [ShLotDw[ i : i+len(ShLotDw)//3 ] for i in range(0, len(ShLotDw), len(ShLotDw)//3)]
    
    splited_SitS = [SitS[ i : i+len(SitS)//6 ] for i in range(0, len(SitS), len(SitS)//6)]
    splited_SitR = [SitR[ i : i+len(SitR)//6 ] for i in range(0, len(SitR), len(SitR)//6)]
    dic_IDs = {}
    # dic_IDs['IDs'] = np.array(all_IDs)
    dic_IDs['ShLotUp1_IDs'] = np.array(splited_ShLotUp[0])
    dic_IDs['ShLotUp2_IDs'] = np.array(splited_ShLotUp[1])
    dic_IDs['ShLotUp3_IDs'] = np.array(splited_ShLotUp[2])
    # dic_IDs['ShLotUp4_IDs'] = np.array(splited_ShLotUp[3])
    
    dic_IDs['ShLotDw1_IDs'] = np.array(splited_ShLotDw[0])
    dic_IDs['ShLotDw2_IDs'] = np.array(splited_ShLotDw[1])
    dic_IDs['ShLotDw3_IDs'] = np.array(splited_ShLotDw[2])
    # dic_IDs['ShLotDw4_IDs'] = np.array(splited_ShLotDw[3])
    
    dic_IDs['SitS1_IDs'] = np.array(splited_SitS[0])
    dic_IDs['SitS2_IDs'] = np.array(splited_SitS[1])
    dic_IDs['SitS3_IDs'] = np.array(splited_SitS[2])
    dic_IDs['SitS4_IDs'] = np.array(splited_SitS[3])
    dic_IDs['SitS5_IDs'] = np.array(splited_SitS[4])
    dic_IDs['SitS6_IDs'] = np.array(splited_SitS[5])
    
    dic_IDs['SitR1_IDs'] = np.array(splited_SitR[0])
    dic_IDs['SitR2_IDs'] = np.array(splited_SitR[1])
    dic_IDs['SitR3_IDs'] = np.array(splited_SitR[2])
    dic_IDs['SitR4_IDs'] = np.array(splited_SitR[3])
    dic_IDs['SitR5_IDs'] = np.array(splited_SitR[4])
    dic_IDs['SitR6_IDs'] = np.array(splited_SitR[5])
    
    np.savez_compressed('monoIDs_'+label_type+'_'+str(num_ds_row)+'_'+str(num_ds_col)+'_'+future_type+'_'+features_type+'.npz', **dic_IDs)
    
    print('available : ')
    print(done_symbols)
    print('# of done_symbols : ' + str(len(done_symbols)))
    print('not available : ')
    notAvailable = list(set(SymbolsUSDT) - set(done_symbols))
    print(notAvailable)
    
    print('Short_Lot_Up1 : ' + str(len(splited_ShLotUp[0])))
    print('Short_Lot_Up2 : ' + str(len(splited_ShLotUp[1])))
    print('Short_Lot_Up3 : ' + str(len(splited_ShLotUp[2])))
    # print('Short_Lot_Up4 : ' + str(len(splited_ShLotUp[3])))
    
    print('Short_Lot_Down1 : ' + str(len(splited_ShLotDw[0])))
    print('Short_Lot_Down2 : ' + str(len(splited_ShLotDw[1])))
    print('Short_Lot_Down3 : ' + str(len(splited_ShLotDw[2])))
    # print('Short_Lot_Down4 : ' + str(len(splited_ShLotDw[3])))
    
    print('Sit_for_support1 : ' + str(len(splited_SitS[0])))
    print('Sit_for_support2 : ' + str(len(splited_SitS[1])))
    print('Sit_for_support3 : ' + str(len(splited_SitS[2])))
    print('Sit_for_support4 : ' + str(len(splited_SitS[3])))
    print('Sit_for_support5 : ' + str(len(splited_SitS[4])))
    print('Sit_for_support6 : ' + str(len(splited_SitS[5])))
    
    print('Sit_for_Resistance1 : ' + str(len(splited_SitR[0])))
    print('Sit_for_Resistance2 : ' + str(len(splited_SitR[1])))
    print('Sit_for_Resistance3 : ' + str(len(splited_SitR[2])))
    print('Sit_for_Resistance4 : ' + str(len(splited_SitR[3])))
    print('Sit_for_Resistance5 : ' + str(len(splited_SitR[4])))
    print('Sit_for_Resistance6 : ' + str(len(splited_SitR[5])))