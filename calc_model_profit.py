# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:24:41 2021

@author: Mahdi
"""

import numpy as np
import pandas as pd
import concurrent.futures
import traceback

def get_profit(args):
    try:
        symbol, budget, LUB, LD, S, LUS, init, end = args
        num_ds_row = 96
        c_df_h = pd.read_csv(symbol + '.csv', index_col=0)
        c_df_h = c_df_h.loc[ init:end , : ]
        
        label_profit_budget_same = 0
        label_end_cash_budget_accumilate = budget
        label_bought1_sold0 = 0
        label_bought_price = 0
        label_transaction_num = 0
        
        model_profit_budget_same = 0
        model_end_cash_budget_accumilate = budget
        model_bought1_sold0 = 0
        model_bought_price = 0
        model_transaction_num = 0
        
        for i in range(num_ds_row, len(c_df_h)-25):
            if (label_bought1_sold0 == 0) and (c_df_h.iloc[i]['Labels'] == 'Short_Lot_Up'):
                label_bought1_sold0 = 1
                label_bought_price = c_df_h.iloc[i]['close']
            if (label_bought1_sold0 == 1) and (c_df_h.iloc[i]['Labels'] != 'Short_Lot_Up'):
                label_bought1_sold0 = 0
                label_profit_budget_same += 0.9985*(1.0 + ((c_df_h.iloc[i]['close'] - label_bought_price) / label_bought_price)) - 1.0
                label_end_cash_budget_accumilate *= 0.9985*(1.0 + ((c_df_h.iloc[i]['close'] - label_bought_price)/label_bought_price))
                label_transaction_num += 1
        
            if (model_bought1_sold0 == 0) and (
                    c_df_h.iloc[i]['model_prediction'] == 'Short_Lot_Up' and c_df_h.iloc[i]['LU'] >= LUB):
                model_bought1_sold0 = 1
                model_bought_price = c_df_h.iloc[i]['close']
            if (model_bought1_sold0 == 1) and c_df_h.iloc[i]['LU'] <= LUS and (
                    (c_df_h.iloc[i]['model_prediction'] == 'Short_Lot_Down' and c_df_h.iloc[i]['LD'] >= LD) or (c_df_h.iloc[i]['model_prediction'] == 'Sit' and c_df_h.iloc[i]['S'] >= S)):
                model_bought1_sold0 = 0
                model_profit_budget_same += 0.9985*(1.0 + ((c_df_h.iloc[i]['close'] - model_bought_price) / model_bought_price)) - 1.0
                model_end_cash_budget_accumilate *= 0.9985*(1.0 + ((c_df_h.iloc[i]['close'] - model_bought_price)/model_bought_price))
                model_transaction_num += 1
        
        hold_end = 0.9985*(1.0 + ((c_df_h.iloc[len(c_df_h)-25]['close'] - c_df_h.iloc[num_ds_row]['close']) / c_df_h.iloc[num_ds_row]['close'])) - 1.0
        
        profits = [round(label_profit_budget_same, 4), round(model_profit_budget_same, 4), round(label_end_cash_budget_accumilate, 4), round(model_end_cash_budget_accumilate, 4), label_transaction_num, model_transaction_num, round(hold_end, 4)]
        # profits = [label_profit_budget_same, label_end_cash_budget_accumilate, label_transaction_num, hold_end]
        
        return symbol, profits
    except:
        # printing stack trace
        traceback.print_exc()
        return symbol, 'ZZZ'

if __name__ == '__main__':
    num_ds_row = 84
    num_ds_col = 87
    future_type = '5mto6h'
    label_type = '3Up3Dw6SS6SR_8&10'
    features_type = 'wr&roc_nosqr'
    num_label_type = 3 + 3 + 6 + 6
    filters = 84
    kernel_size = 6
    model_architecture = '3Conv1Df'+str(filters)+'k'+str(kernel_size)+'psame_MxPl1Do2_Dense128'
    DNN_activation = 'relu'
    last_activation = 'softmax'
    
    batch_size = 2 * num_label_type
    weights_path = None
    
    experiment_name = label_type+'_'+str(batch_size)+'_'+str(num_ds_row)+'_'+str(num_ds_col)+'_'+future_type+'_'+features_type+'_'+model_architecture+'_'+DNN_activation+'_'+last_activation
    

    trade_actions = ['Short_Lot_Up', 'Sit', 'Short_Lot_Down']
    nb_actions = 3
    
    budget = 1
    LUB = 0.60
    LD = 0.5
    S = 0.5
    LUS = 0.05
    
    # SymbolsUSDT = ['1INCHUSDT', 'AAVEUSDT', 'ACMUSDT', 'ADAUSDT', 'AIONUSDT', 'AKROUSDT', 'ALGOUSDT', 'ALPHAUSDT', 'ANKRUSDT', 'ANTUSDT', 'ARDRUSDT', 'ARPAUSDT', 'ASRUSDT', 'ATMUSDT', 'ATOMUSDT', 'AUDIOUSDT', 'AUDUSDT', 'AVAUSDT', 'AVAXUSDT', 'AXSUSDT', 'BADGERUSDT', 'BALUSDT', 'BANDUSDT', 'BATUSDT', 'BCHUSDT', 'BEAMUSDT', 'BELUSDT', 'BLZUSDT', 'BNBUSDT', 'BNTUSDT', 'BTCUSDT', 'BTSUSDT', 'BTTUSDT', 'BZRXUSDT', 'CAKEUSDT', 'CELOUSDT', 'CELRUSDT', 'CHRUSDT', 'CHZUSDT', 'CKBUSDT', 'COCOSUSDT', 'COMPUSDT', 'COSUSDT', 'COTIUSDT', 'CRVUSDT', 'CTKUSDT', 'CTSIUSDT', 'CTXCUSDT', 'CVCUSDT', 'DASHUSDT', 'DATAUSDT', 'DCRUSDT', 'DENTUSDT', 'DGBUSDT', 'DIAUSDT', 'DNTUSDT', 'DOCKUSDT', 'DODOUSDT', 'DOGEUSDT', 'DOTUSDT', 'DREPUSDT', 'DUSKUSDT', 'EGLDUSDT', 'ENJUSDT', 'EOSUSDT', 'ETCUSDT', 'ETHUSDT', 'FETUSDT', 'FILUSDT', 'FIOUSDT', 'FIROUSDT', 'FISUSDT', 'FLMUSDT', 'FTMUSDT', 'FTTUSDT', 'FUNUSDT', 'GRTUSDT', 'GTOUSDT', 'GXSUSDT', 'HARDUSDT', 'HBARUSDT', 'HIVEUSDT', 'HNTUSDT', 'HOTUSDT', 'ICXUSDT', 'INJUSDT', 'IOSTUSDT', 'IOTAUSDT', 'IOTXUSDT', 'IRISUSDT', 'JSTUSDT', 'JUVUSDT', 'KAVAUSDT', 'KEYUSDT', 'KMDUSDT', 'KNCUSDT', 'KSMUSDT', 'LINKUSDT', 'LITUSDT', 'LRCUSDT', 'LSKUSDT', 'LTCUSDT', 'LTOUSDT', 'LUNAUSDT', 'MANAUSDT', 'MATICUSDT', 'MBLUSDT', 'MDTUSDT', 'MFTUSDT', 'MITHUSDT', 'MKRUSDT', 'MTLUSDT', 'NANOUSDT', 'NBSUSDT', 'NEARUSDT', 'NEOUSDT', 'NKNUSDT', 'NMRUSDT', 'NPXSUSDT', 'NULSUSDT', 'OCEANUSDT', 'OGNUSDT', 'OMGUSDT', 'ONEUSDT', 'ONGUSDT', 'ONTUSDT', 'ORNUSDT', 'OXTUSDT', 'PAXGUSDT', 'PAXUSDT', 'PERLUSDT', 'PNTUSDT', 'PSGUSDT', 'QTUMUSDT', 'REEFUSDT', 'RENUSDT', 'REPUSDT', 'RIFUSDT', 'RLCUSDT', 'ROSEUSDT', 'RSRUSDT', 'RUNEUSDT', 'RVNUSDT', 'SANDUSDT', 'SFPUSDT', 'SKLUSDT', 'SNXUSDT', 'SOLUSDT', 'SRMUSDT', 'STMXUSDT', 'STORJUSDT', 'STPTUSDT', 'STRAXUSDT', 'STXUSDT', 'SUNUSDT', 'SUSDUSDT', 'SUSHIUSDT', 'SXPUSDT', 'TCTUSDT', 'TFUELUSDT', 'THETAUSDT', 'TOMOUSDT', 'TRBUSDT', 'TROYUSDT', 'TRUUSDT', 'TRXUSDT', 'TWTUSDT', 'UMAUSDT', 'UNFIUSDT', 'UNIUSDT', 'UTKUSDT', 'VETUSDT', 'VITEUSDT', 'VTHOUSDT', 'WANUSDT', 'WAVESUSDT', 'WINGUSDT', 'WINUSDT', 'WNXMUSDT', 'WRXUSDT', 'WTCUSDT', 'XEMUSDT', 'XLMUSDT', 'XMRUSDT', 'XRPUSDT', 'XTZUSDT', 'XVSUSDT', 'YFIIUSDT', 'YFIUSDT', 'ZECUSDT', 'ZENUSDT', 'ZILUSDT', 'ZRXUSDT']
    SymbolsUSDT = ['ALPHAUSDT', 'MATICUSDT', 'LINKUSDT', 'XRPUSDT', 'BADGERUSDT']
    print(SymbolsUSDT)
    
    print('budget : '+str(budget))
    print(experiment_name)
    # print('Since Nov 2nd till May 22nd'+' LUB>{0} LD>{1} S>{2} LUS<{3}'.format(LUB, LD, S, LUS))
    print('Since June 1st till end'+' LUB>{0} LD>{1} S>{2} LUS<{3}'.format(LUB, LD, S, LUS))
    
    init = {}
    end = {}
    for symbol in SymbolsUSDT:
        # # since Nov 2nd till May 22nd
        # init[symbol] = '2020-11-02 00:35:00'
        # end[symbol] = '2021-05-22 17:50:00'
        # Since June 1st till end
        init[symbol] = '2021-06-01 00:00:00'
        end[symbol] = '2021-06-20 17:00:00'
    
    # init['ALPHAUSDT'] = '2020-11-02 00:35:00'
    # init['MATICUSDT'] = '2020-11-02 00:35:00'
    # init['LINKUSDT'] = '2020-11-02 00:35:00'
    # init['XRPUSDT'] = '2020-11-02 00:35:00'
    init['BADGERUSDT'] = '2021-03-09 00:40:00'
    # end['ALPHAUSDT'] = '2021-05-22 17:50:00'
    # end['MATICUSDT'] = '2021-05-22 18:35:00'
    # end['LINKUSDT'] = '2021-05-22 18:35:00'
    # end['XRPUSDT'] = '2021-05-22 18:40:00'
    # end['BADGERUSDT'] = '2021-05-22 17:50:00'
    
    args = ((symbol, budget, LUB, LD, S, LUS, init[symbol], end[symbol]) for symbol in SymbolsUSDT)
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        results = executor.map(get_profit, args)
    for symbol, profits in results:
        if profits != 'ZZZ':
            print('\nSymbol: ' + symbol)
            print('end_cash_budget_same :\n'+'label:'+str(profits[0]*budget)+'  model: '+str(profits[1]*budget)+' proportion: '+str(round(profits[1]/profits[0], 4)))
            print('end_cash_budget_accumilate :\n'+'label: '+str(profits[2])+'  model: '+str(profits[3])+' proportion: '+str(round((profits[3]-1)/(profits[2]-1), 4)))
            print('num_of_transactions :\n'+'label: '+str(profits[4])+'  model: '+str(profits[5])+' proportion: '+str(round(profits[5]/profits[4], 4)))
            print('hold_till_end_cash :\n'+str(profits[6]*budget))
            
            # print('end_cash_budget_same :\n'+'label:'+str(profits[0]*budget))
            # print('end_cash_budget_accumilate :\n'+'label: '+str(profits[1]))
            # print('num_of_transactions :\n'+'label: '+str(profits[2]))
            # print('hold_till_end_cash :\n'+str(profits[3]*budget))