from binance.client import Client
from math import floor, log10, tanh, sqrt
from time import sleep
import random
from datetime import timedelta, datetime, timezone
# from binance.helpers import date_to_milliseconds, interval_to_milliseconds
import backtrader as bt
import pandas as pd
import btalib
import numpy as np
import threading
import concurrent.futures
import traceback

import tensorflow as tf
from keras.models import load_model
import os


class KeltnerChannel(bt.Indicator):
    lines = ('mid25', 'upper25', 'lower25', 'mid99', 'upper99', 'lower99')
    params = dict(
            pfast=25,  # period for the fast moving average
            pslow=99,  # period for the slow moving average
            atr_coe=2
    )

    plotinfo = dict(subplot=False)  # plot along with data
    plotlines = dict(
        mid25=dict(ls='--', color='purple'),  # dashed line
        upper25=dict(color='purple'),  # use same color as prev line (mid)
        lower25=dict(color='purple'),  # use same color as prev line (upper)

        mid99=dict(ls='--', color='cyan'),  # dashed line
        upper99=dict(color='cyan'),  # use same color as prev line (mid)
        lower99=dict(color='cyan'),  # use same color as prev line (upper)
    )

    def __init__(self):
        self.l.mid25 = bt.ind.EMA(self.data, period=self.p.pfast)
        self.l.upper25 = self.l.mid25 + bt.ind.ATR(self.data, period=self.p.pfast) * self.p.atr_coe
        self.l.lower25 = self.l.mid25 - bt.ind.ATR(self.data, period=self.p.pfast) * self.p.atr_coe

        self.l.mid99 = bt.ind.EMA(period=self.p.pslow)
        self.l.upper99 = self.l.mid99 + bt.ind.ATR(self.data, period=self.p.pslow) * self.p.atr_coe
        self.l.lower99 = self.l.mid99 - bt.ind.ATR(self.data, period=self.p.pslow) * self.p.atr_coe


class VolumeIndicators(bt.Indicator):
    lines = ('sma', 'numTrade', 'VolPerTrade',)
    params = (('period', 12),
              ('safediv', True),
              ('safehigh', 0.0),
              ('safelow', 0.0),)
    def __init__(self):
        self.l.sma = bt.ind.SMA(self.data.QuoteAssetVolume, period=self.p.period)
        self.l.numTrade = bt.ind.SumN(self.data.NumOfTrades, period=self.p.period)
        if not self.p.safediv:
            vpt = self.l.sma * self.p.period / self.l.numTrade
        else:
            highrs = self.p.safehigh
            lowrs = self.p.safelow
            vpt = bt.functions.DivZeroByZero(self.l.sma * self.p.period, self.l.numTrade, highrs, lowrs)
        self.l.VolPerTrade = vpt


class BuySellDiff(bt.Indicator):
    lines = ('buy_volSum', 'sell_volSum', 'diff',)
    params = (('period', 12),)
    def __init__(self):
        self.l.buy_volSum = bt.ind.SumN(self.data.TakerBuyQuoteAssetVolume, period=self.p.period)
        self.l.sell_volSum = bt.ind.SumN(self.data.TakerSellQuoteAssetVolume, period=self.p.period)
        self.l.diff = bt.ind.MYOscillator(self.l.buy_volSum, self.l.sell_volSum)


class BarAnalysis(bt.analyzers.Analyzer):
    def start(self):
        self.rets = []

    def next(self):
        try:
            open = round(self.datas[0].open[0], abs(min(0, int(floor(log10(self.datas[0].open[0])) - 3))))
            high = round(self.datas[0].high[0], abs(min(0, int(floor(log10(self.datas[0].high[0])) - 3))))
            low = round(self.datas[0].low[0], abs(min(0, int(floor(log10(self.datas[0].low[0])) - 3))))
            close = round(self.datas[0].close[0], abs(min(0, int(floor(log10(self.datas[0].close[0])) - 3))))

            keltner_5m_upper25 = self.strategy.keltner_5m.l.upper25[0]
            keltner_5m_upper25 = round(keltner_5m_upper25, abs(min(0, int(floor(log10(abs(keltner_5m_upper25))) - 3))))
            keltner_5m_mid25 = self.strategy.keltner_5m.l.mid25[0]
            keltner_5m_mid25 = round(keltner_5m_mid25, abs(min(0, int(floor(log10(abs(keltner_5m_mid25))) - 3))))
            keltner_5m_lower25 = self.strategy.keltner_5m.l.lower25[0]
            keltner_5m_lower25 = round(keltner_5m_lower25, abs(min(0, int(floor(log10(abs(keltner_5m_lower25))) - 3))))

            keltner_5m_upper99 = self.strategy.keltner_5m.l.upper99[0]
            keltner_5m_upper99 = round(keltner_5m_upper99, abs(min(0, int(floor(log10(abs(keltner_5m_upper99))) - 3))))
            keltner_5m_mid99 = self.strategy.keltner_5m.l.mid99[0]
            keltner_5m_mid99 = round(keltner_5m_mid99, abs(min(0, int(floor(log10(abs(keltner_5m_mid99))) - 3))))
            keltner_5m_lower99 = self.strategy.keltner_5m.l.lower99[0]
            keltner_5m_lower99 = round(keltner_5m_lower99, abs(min(0, int(floor(log10(abs(keltner_5m_lower99))) - 3))))

            keltner_1h_upper25 = self.strategy.keltner_1h.l.upper25[0]
            keltner_1h_upper25 = round(keltner_1h_upper25, abs(min(0, int(floor(log10(abs(keltner_1h_upper25))) - 3))))
            keltner_1h_mid25 = self.strategy.keltner_1h.l.mid25[0]
            keltner_1h_mid25 = round(keltner_1h_mid25, abs(min(0, int(floor(log10(abs(keltner_1h_mid25))) - 3))))
            keltner_1h_lower25 = self.strategy.keltner_1h.l.lower25[0]
            keltner_1h_lower25 = round(keltner_1h_lower25, abs(min(0, int(floor(log10(abs(keltner_1h_lower25))) - 3))))

            keltner_1h_upper99 = self.strategy.keltner_1h.l.upper99[0]
            keltner_1h_upper99 = round(keltner_1h_upper99, abs(min(0, int(floor(log10(abs(keltner_1h_upper99))) - 3))))
            keltner_1h_mid99 = self.strategy.keltner_1h.l.mid99[0]
            keltner_1h_mid99 = round(keltner_1h_mid99, abs(min(0, int(floor(log10(abs(keltner_1h_mid99))) - 3))))
            keltner_1h_lower99 = self.strategy.keltner_1h.l.lower99[0]
            keltner_1h_lower99 = round(keltner_1h_lower99, abs(min(0, int(floor(log10(abs(keltner_1h_lower99))) - 3))))

            fib_day_s3 = self.strategy.fib_day.l.s3[0]
            fib_day_s3 = round(fib_day_s3, abs(min(0, int(floor(log10(abs(fib_day_s3))) - 3))))
            fib_day_s2 = self.strategy.fib_day.l.s2[0]
            fib_day_s2 = round(fib_day_s2, abs(min(0, int(floor(log10(abs(fib_day_s2))) - 3))))
            fib_day_s1 = self.strategy.fib_day.l.s1[0]
            fib_day_s1 = round(fib_day_s1, abs(min(0, int(floor(log10(abs(fib_day_s1))) - 3))))
            fib_day_p = self.strategy.fib_day.l.p[0]
            fib_day_p = round(fib_day_p, abs(min(0, int(floor(log10(abs(fib_day_p))) - 3))))
            fib_day_r1 = self.strategy.fib_day.l.r1[0]
            fib_day_r1 = round(fib_day_r1, abs(min(0, int(floor(log10(abs(fib_day_r1))) - 3))))
            fib_day_r2 = self.strategy.fib_day.l.r2[0]
            fib_day_r2 = round(fib_day_r2, abs(min(0, int(floor(log10(abs(fib_day_r2))) - 3))))
            fib_day_r3 = self.strategy.fib_day.l.r3[0]
            fib_day_r3 = round(fib_day_r3, abs(min(0, int(floor(log10(abs(fib_day_r3))) - 3))))

            trad_day_s5 = self.strategy.trad_day.l.s5[0]
            trad_day_s5 = round(trad_day_s5, abs(min(0, int(floor(log10(abs(trad_day_s5))) - 3))))
            trad_day_s4 = self.strategy.trad_day.l.s4[0]
            trad_day_s4 = round(trad_day_s4, abs(min(0, int(floor(log10(abs(trad_day_s4))) - 3))))
            trad_day_s3 = self.strategy.trad_day.l.s3[0]
            trad_day_s3 = round(trad_day_s3, abs(min(0, int(floor(log10(abs(trad_day_s3))) - 3))))
            trad_day_s2 = self.strategy.trad_day.l.s2[0]
            trad_day_s2 = round(trad_day_s2, abs(min(0, int(floor(log10(abs(trad_day_s2))) - 3))))
            trad_day_s1 = self.strategy.trad_day.l.s1[0]
            trad_day_s1 = round(trad_day_s1, abs(min(0, int(floor(log10(abs(trad_day_s1))) - 3))))
            trad_day_p = self.strategy.trad_day.l.p[0]
            trad_day_p = round(trad_day_p, abs(min(0, int(floor(log10(abs(trad_day_p))) - 3))))
            trad_day_r1 = self.strategy.trad_day.l.r1[0]
            trad_day_r1 = round(trad_day_r1, abs(min(0, int(floor(log10(abs(trad_day_r1))) - 3))))
            trad_day_r2 = self.strategy.trad_day.l.r2[0]
            trad_day_r2 = round(trad_day_r2, abs(min(0, int(floor(log10(abs(trad_day_r2))) - 3))))
            trad_day_r3 = self.strategy.trad_day.l.r3[0]
            trad_day_r3 = round(trad_day_r3, abs(min(0, int(floor(log10(abs(trad_day_r3))) - 3))))
            trad_day_r4 = self.strategy.trad_day.l.r4[0]
            trad_day_r4 = round(trad_day_r4, abs(min(0, int(floor(log10(abs(trad_day_r4))) - 3))))
            trad_day_r5 = self.strategy.trad_day.l.r5[0]
            trad_day_r5 = round(trad_day_r5, abs(min(0, int(floor(log10(abs(trad_day_r5))) - 3))))

            dem_day_s1 = self.strategy.dem_day.l.s1[0]
            dem_day_s1 = round(dem_day_s1, abs(min(0, int(floor(log10(abs(dem_day_s1))) - 3))))
            dem_day_p = self.strategy.dem_day.l.p[0]
            dem_day_p = round(dem_day_p, abs(min(0, int(floor(log10(abs(dem_day_p))) - 3))))
            dem_day_r1 = self.strategy.dem_day.l.r1[0]
            dem_day_r1 = round(dem_day_r1, abs(min(0, int(floor(log10(abs(dem_day_r1))) - 3))))

            fib_week_s3 = self.strategy.fib_week.l.s3[0]
            fib_week_s3 = round(fib_week_s3, abs(min(0, int(floor(log10(abs(fib_week_s3))) - 3))))
            fib_week_s2 = self.strategy.fib_week.l.s2[0]
            fib_week_s2 = round(fib_week_s2, abs(min(0, int(floor(log10(abs(fib_week_s2))) - 3))))
            fib_week_s1 = self.strategy.fib_week.l.s1[0]
            fib_week_s1 = round(fib_week_s1, abs(min(0, int(floor(log10(abs(fib_week_s1))) - 3))))
            fib_week_p = self.strategy.fib_week.l.p[0]
            fib_week_p = round(fib_week_p, abs(min(0, int(floor(log10(abs(fib_week_p))) - 3))))
            fib_week_r1 = self.strategy.fib_week.l.r1[0]
            fib_week_r1 = round(fib_week_r1, abs(min(0, int(floor(log10(abs(fib_week_r1))) - 3))))
            fib_week_r2 = self.strategy.fib_week.l.r2[0]
            fib_week_r2 = round(fib_week_r2, abs(min(0, int(floor(log10(abs(fib_week_r2))) - 3))))
            fib_week_r3 = self.strategy.fib_week.l.r3[0]
            fib_week_r3 = round(fib_week_r3, abs(min(0, int(floor(log10(abs(fib_week_r3))) - 3))))

            trad_week_s5 = self.strategy.trad_week.l.s5[0]
            trad_week_s5 = round(trad_week_s5, abs(min(0, int(floor(log10(abs(trad_week_s5))) - 3))))
            trad_week_s4 = self.strategy.trad_week.l.s4[0]
            trad_week_s4 = round(trad_week_s4, abs(min(0, int(floor(log10(abs(trad_week_s4))) - 3))))
            trad_week_s3 = self.strategy.trad_week.l.s3[0]
            trad_week_s3 = round(trad_week_s3, abs(min(0, int(floor(log10(abs(trad_week_s3))) - 3))))
            trad_week_s2 = self.strategy.trad_week.l.s2[0]
            trad_week_s2 = round(trad_week_s2, abs(min(0, int(floor(log10(abs(trad_week_s2))) - 3))))
            trad_week_s1 = self.strategy.trad_week.l.s1[0]
            trad_week_s1 = round(trad_week_s1, abs(min(0, int(floor(log10(abs(trad_week_s1))) - 3))))
            trad_week_p = self.strategy.trad_week.l.p[0]
            trad_week_p = round(trad_week_p, abs(min(0, int(floor(log10(abs(trad_week_p))) - 3))))
            trad_week_r1 = self.strategy.trad_week.l.r1[0]
            trad_week_r1 = round(trad_week_r1, abs(min(0, int(floor(log10(abs(trad_week_r1))) - 3))))
            trad_week_r2 = self.strategy.trad_week.l.r2[0]
            trad_week_r2 = round(trad_week_r2, abs(min(0, int(floor(log10(abs(trad_week_r2))) - 3))))
            trad_week_r3 = self.strategy.trad_week.l.r3[0]
            trad_week_r3 = round(trad_week_r3, abs(min(0, int(floor(log10(abs(trad_week_r3))) - 3))))
            trad_week_r4 = self.strategy.trad_week.l.r4[0]
            trad_week_r4 = round(trad_week_r4, abs(min(0, int(floor(log10(abs(trad_week_r4))) - 3))))
            trad_week_r5 = self.strategy.trad_week.l.r5[0]
            trad_week_r5 = round(trad_week_r5, abs(min(0, int(floor(log10(abs(trad_week_r5))) - 3))))

            dem_week_s1 = self.strategy.dem_week.l.s1[0]
            dem_week_s1 = round(dem_week_s1, abs(min(0, int(floor(log10(abs(dem_week_s1))) - 3))))
            dem_week_p = self.strategy.dem_week.l.p[0]
            dem_week_p = round(dem_week_p, abs(min(0, int(floor(log10(abs(dem_week_p))) - 3))))
            dem_week_r1 = self.strategy.dem_week.l.r1[0]
            dem_week_r1 = round(dem_week_r1, abs(min(0, int(floor(log10(abs(dem_week_r1))) - 3))))

            cls_keltner_5m_lower25 = 0
            cls_keltner_5m_mid25 = 0
            cls_keltner_5m_upper25 = 0
            if close < keltner_5m_lower25:
                temp = round((close - keltner_5m_lower25) / keltner_5m_lower25, 4)
                cls_keltner_5m_lower25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
            elif keltner_5m_lower25 <= close < keltner_5m_mid25:
                temp = round((close - keltner_5m_lower25) / keltner_5m_lower25, 4)
                cls_keltner_5m_lower25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
                temp = round((close - keltner_5m_mid25) / keltner_5m_mid25, 4)
                cls_keltner_5m_mid25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
            elif keltner_5m_mid25 <= close < keltner_5m_upper25:
                temp = round((close - keltner_5m_mid25) / keltner_5m_mid25, 4)
                cls_keltner_5m_mid25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
                temp = round((close - keltner_5m_upper25) / keltner_5m_upper25, 4)
                cls_keltner_5m_upper25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
            elif keltner_5m_upper25 <= close:
                temp = round((close - keltner_5m_upper25) / keltner_5m_upper25, 4)
                cls_keltner_5m_upper25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)

            cls_keltner_5m_lower99 = 0
            cls_keltner_5m_mid99 = 0
            cls_keltner_5m_upper99 = 0
            if close < keltner_5m_lower99:
                temp = round((close - keltner_5m_lower99) / keltner_5m_lower99, 4)
                cls_keltner_5m_lower99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
            elif keltner_5m_lower99 <= close < keltner_5m_mid99:
                temp = round((close - keltner_5m_lower99) / keltner_5m_lower99, 4)
                cls_keltner_5m_lower99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
                temp = round((close - keltner_5m_mid99) / keltner_5m_mid99, 4)
                cls_keltner_5m_mid99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
            elif keltner_5m_mid99 <= close < keltner_5m_upper99:
                temp = round((close - keltner_5m_mid99) / keltner_5m_mid99, 4)
                cls_keltner_5m_mid99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
                temp = round((close - keltner_5m_upper99) / keltner_5m_upper99, 4)
                cls_keltner_5m_upper99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
            elif keltner_5m_upper99 <= close:
                temp = round((close - keltner_5m_upper99) / keltner_5m_upper99, 4)
                cls_keltner_5m_upper99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)

            cls_keltner_1h_lower25 = 0
            cls_keltner_1h_mid25 = 0
            cls_keltner_1h_upper25 = 0
            if close < keltner_1h_lower25:
                temp = round((close - keltner_1h_lower25) / keltner_1h_lower25, 4)
                cls_keltner_1h_lower25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
            elif keltner_1h_lower25 <= close < keltner_1h_mid25:
                temp = round((close - keltner_1h_lower25) / keltner_1h_lower25, 4)
                cls_keltner_1h_lower25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
                temp = round((close - keltner_1h_mid25) / keltner_1h_mid25, 4)
                cls_keltner_1h_mid25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
            elif keltner_1h_mid25 <= close < keltner_1h_upper25:
                temp = round((close - keltner_1h_mid25) / keltner_1h_mid25, 4)
                cls_keltner_1h_mid25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
                temp = round((close - keltner_1h_upper25) / keltner_1h_upper25, 4)
                cls_keltner_1h_upper25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
            elif keltner_1h_upper25 <= close:
                temp = round((close - keltner_1h_upper25) / keltner_1h_upper25, 4)
                cls_keltner_1h_upper25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)

            cls_keltner_1h_lower99 = 0
            cls_keltner_1h_mid99 = 0
            cls_keltner_1h_upper99 = 0
            if close < keltner_1h_lower99:
                temp = round((close - keltner_1h_lower99) / keltner_1h_lower99, 4)
                cls_keltner_1h_lower99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
            elif keltner_1h_lower99 <= close < keltner_1h_mid99:
                temp = round((close - keltner_1h_lower99) / keltner_1h_lower99, 4)
                cls_keltner_1h_lower99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
                temp = round((close - keltner_1h_mid99) / keltner_1h_mid99, 4)
                cls_keltner_1h_mid99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
            elif keltner_1h_mid99 <= close < keltner_1h_upper99:
                temp = round((close - keltner_1h_mid99) / keltner_1h_mid99, 4)
                cls_keltner_1h_mid99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
                temp = round((close - keltner_1h_upper99) / keltner_1h_upper99, 4)
                cls_keltner_1h_upper99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
            elif keltner_1h_upper99 <= close:
                temp = round((close - keltner_1h_upper99) / keltner_1h_upper99, 4)
                cls_keltner_1h_upper99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)

            cls_fib_day_s3 = 0
            cls_fib_day_s2 = 0
            cls_fib_day_s1 = 0
            cls_fib_day_p = 0
            cls_fib_day_r1 = 0
            cls_fib_day_r2 = 0
            cls_fib_day_r3 = 0
            if close < fib_day_s3:
                temp = round((close - fib_day_s3) / fib_day_s3, 4)
                cls_fib_day_s3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif fib_day_s3 <= close < fib_day_s2:
                temp = round((close - fib_day_s3) / fib_day_s3, 4)
                cls_fib_day_s3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - fib_day_s2) / fib_day_s2, 4)
                cls_fib_day_s2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif fib_day_s2 <= close < fib_day_s1:
                temp = round((close - fib_day_s2) / fib_day_s2, 4)
                cls_fib_day_s2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - fib_day_s1) / fib_day_s1, 4)
                cls_fib_day_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif fib_day_s1 <= close < fib_day_p:
                temp = round((close - fib_day_s1) / fib_day_s1, 4)
                cls_fib_day_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - fib_day_p) / fib_day_p, 4)
                cls_fib_day_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif fib_day_p <= close < fib_day_r1:
                temp = round((close - fib_day_p) / fib_day_p, 4)
                cls_fib_day_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - fib_day_r1) / fib_day_r1, 4)
                cls_fib_day_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif fib_day_r1 <= close < fib_day_r2:
                temp = round((close - fib_day_r1) / fib_day_r1, 4)
                cls_fib_day_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - fib_day_r2) / fib_day_r2, 4)
                cls_fib_day_r2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif fib_day_r2 <= close < fib_day_r3:
                temp = round((close - fib_day_r2) / fib_day_r2, 4)
                cls_fib_day_r2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - fib_day_r3) / fib_day_r3, 4)
                cls_fib_day_r3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif fib_day_r3 <= close:
                temp = round((close - fib_day_r3) / fib_day_r3, 4)
                cls_fib_day_r3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)

            cls_trad_day_s5 = 0
            cls_trad_day_s4 = 0
            cls_trad_day_s3 = 0
            cls_trad_day_s2 = 0
            cls_trad_day_s1 = 0
            cls_trad_day_p = 0
            cls_trad_day_r1 = 0
            cls_trad_day_r2 = 0
            cls_trad_day_r3 = 0
            cls_trad_day_r4 = 0
            cls_trad_day_r5 = 0
            if close < trad_day_s5:
                temp = round((close - trad_day_s5) / trad_day_s5, 4)
                cls_trad_day_s5 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_day_s5 <= close < trad_day_s4:
                temp = round((close - trad_day_s5) / trad_day_s5, 4)
                cls_trad_day_s5 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - trad_day_s4) / trad_day_s4, 4)
                cls_trad_day_s4 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_day_s4 <= close < trad_day_s3:
                temp = round((close - trad_day_s4) / trad_day_s4, 4)
                cls_trad_day_s4 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - trad_day_s3) / trad_day_s3, 4)
                cls_trad_day_s3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_day_s3 <= close < trad_day_s2:
                temp = round((close - trad_day_s3) / trad_day_s3, 4)
                cls_trad_day_s3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - trad_day_s2) / trad_day_s2, 4)
                cls_trad_day_s2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_day_s2 <= close < trad_day_s1:
                temp = round((close - trad_day_s2) / trad_day_s2, 4)
                cls_trad_day_s2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - trad_day_s1) / trad_day_s1, 4)
                cls_trad_day_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_day_s1 <= close < trad_day_p:
                temp = round((close - trad_day_s1) / trad_day_s1, 4)
                cls_trad_day_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - trad_day_p) / trad_day_p, 4)
                cls_trad_day_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_day_p <= close < trad_day_r1:
                temp = round((close - trad_day_p) / trad_day_p, 4)
                cls_trad_day_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - trad_day_r1) / trad_day_r1, 4)
                cls_trad_day_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_day_r1 <= close < trad_day_r2:
                temp = round((close - trad_day_r1) / trad_day_r1, 4)
                cls_trad_day_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - trad_day_r2) / trad_day_r2, 4)
                cls_trad_day_r2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_day_r2 <= close < trad_day_r3:
                temp = round((close - trad_day_r2) / trad_day_r2, 4)
                cls_trad_day_r2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - trad_day_r3) / trad_day_r3, 4)
                cls_trad_day_r3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_day_r3 <= close < trad_day_r4:
                temp = round((close - trad_day_r3) / trad_day_r3, 4)
                cls_trad_day_r3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - trad_day_r4) / trad_day_r4, 4)
                cls_trad_day_r4 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_day_r4 <= close < trad_day_r5:
                temp = round((close - trad_day_r4) / trad_day_r4, 4)
                cls_trad_day_r4 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - trad_day_r5) / trad_day_r5, 4)
                cls_trad_day_r5 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_day_r5 <= close:
                temp = round((close - trad_day_r5) / trad_day_r5, 4)
                cls_trad_day_r5 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)

            cls_dem_day_s1 = 0
            cls_dem_day_p = 0
            cls_dem_day_r1 = 0
            if close < dem_day_s1:
                temp = round((close - dem_day_s1) / dem_day_s1, 4)
                cls_dem_day_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif dem_day_s1 <= close < dem_day_p:
                temp = round((close - dem_day_s1) / dem_day_s1, 4)
                cls_dem_day_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - dem_day_p) / dem_day_p, 4)
                cls_dem_day_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif dem_day_p <= close < dem_day_r1:
                temp = round((close - dem_day_p) / dem_day_p, 4)
                cls_dem_day_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - dem_day_r1) / dem_day_r1, 4)
                cls_dem_day_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif dem_day_r1 <= close:
                temp = round((close - dem_day_r1) / dem_day_r1, 4)
                cls_dem_day_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)

            cls_fib_week_s3 = 0
            cls_fib_week_s2 = 0
            cls_fib_week_s1 = 0
            cls_fib_week_p = 0
            cls_fib_week_r1 = 0
            cls_fib_week_r2 = 0
            cls_fib_week_r3 = 0
            if close < fib_week_s3:
                temp = round((close - fib_week_s3) / fib_week_s3, 4)
                cls_fib_week_s3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif fib_week_s3 <= close < fib_week_s2:
                temp = round((close - fib_week_s3) / fib_week_s3, 4)
                cls_fib_week_s3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - fib_week_s2) / fib_week_s2, 4)
                cls_fib_week_s2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif fib_week_s2 <= close < fib_week_s1:
                temp = round((close - fib_week_s2) / fib_week_s2, 4)
                cls_fib_week_s2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - fib_week_s1) / fib_week_s1, 4)
                cls_fib_week_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif fib_week_s1 <= close < fib_week_p:
                temp = round((close - fib_week_s1) / fib_week_s1, 4)
                cls_fib_week_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - fib_week_p) / fib_week_p, 4)
                cls_fib_week_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif fib_week_p <= close < fib_week_r1:
                temp = round((close - fib_week_p) / fib_week_p, 4)
                cls_fib_week_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - fib_week_r1) / fib_week_r1, 4)
                cls_fib_week_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif fib_week_r1 <= close < fib_week_r2:
                temp = round((close - fib_week_r1) / fib_week_r1, 4)
                cls_fib_week_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - fib_week_r2) / fib_week_r2, 4)
                cls_fib_week_r2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif fib_week_r2 <= close < fib_week_r3:
                temp = round((close - fib_week_r2) / fib_week_r2, 4)
                cls_fib_week_r2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - fib_week_r3) / fib_week_r3, 4)
                cls_fib_week_r3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif fib_week_r3 <= close:
                temp = round((close - fib_week_r3) / fib_week_r3, 4)
                cls_fib_week_r3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)

            cls_trad_week_s5 = 0
            cls_trad_week_s4 = 0
            cls_trad_week_s3 = 0
            cls_trad_week_s2 = 0
            cls_trad_week_s1 = 0
            cls_trad_week_p = 0
            cls_trad_week_r1 = 0
            cls_trad_week_r2 = 0
            cls_trad_week_r3 = 0
            cls_trad_week_r4 = 0
            cls_trad_week_r5 = 0
            if close < trad_week_s5:
                temp = round((close - trad_week_s5) / trad_week_s5, 4)
                cls_trad_week_s5 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_week_s5 <= close < trad_week_s4:
                temp = round((close - trad_week_s5) / trad_week_s5, 4)
                cls_trad_week_s5 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - trad_week_s4) / trad_week_s4, 4)
                cls_trad_week_s4 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_week_s4 <= close < trad_week_s3:
                temp = round((close - trad_week_s4) / trad_week_s4, 4)
                cls_trad_week_s4 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - trad_week_s3) / trad_week_s3, 4)
                cls_trad_week_s3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_week_s3 <= close < trad_week_s2:
                temp = round((close - trad_week_s3) / trad_week_s3, 4)
                cls_trad_week_s3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - trad_week_s2) / trad_week_s2, 4)
                cls_trad_week_s2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_week_s2 <= close < trad_week_s1:
                temp = round((close - trad_week_s2) / trad_week_s2, 4)
                cls_trad_week_s2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - trad_week_s1) / trad_week_s1, 4)
                cls_trad_week_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_week_s1 <= close < trad_week_p:
                temp = round((close - trad_week_s1) / trad_week_s1, 4)
                cls_trad_week_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - trad_week_p) / trad_week_p, 4)
                cls_trad_week_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_week_p <= close < trad_week_r1:
                temp = round((close - trad_week_p) / trad_week_p, 4)
                cls_trad_week_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - trad_week_r1) / trad_week_r1, 4)
                cls_trad_week_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_week_r1 <= close < trad_week_r2:
                temp = round((close - trad_week_r1) / trad_week_r1, 4)
                cls_trad_week_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - trad_week_r2) / trad_week_r2, 4)
                cls_trad_week_r2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_week_r2 <= close < trad_week_r3:
                temp = round((close - trad_week_r2) / trad_week_r2, 4)
                cls_trad_week_r2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - trad_week_r3) / trad_week_r3, 4)
                cls_trad_week_r3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_week_r3 <= close < trad_week_r4:
                temp = round((close - trad_week_r3) / trad_week_r3, 4)
                cls_trad_week_r3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - trad_week_r4) / trad_week_r4, 4)
                cls_trad_week_r4 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_week_r4 <= close < trad_week_r5:
                temp = round((close - trad_week_r4) / trad_week_r4, 4)
                cls_trad_week_r4 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - trad_week_r5) / trad_week_r5, 4)
                cls_trad_week_r5 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif trad_week_r5 <= close:
                temp = round((close - trad_week_r5) / trad_week_r5, 4)
                cls_trad_week_r5 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)

            cls_dem_week_s1 = 0
            cls_dem_week_p = 0
            cls_dem_week_r1 = 0
            if close < dem_week_s1:
                temp = round((close - dem_week_s1) / dem_week_s1, 4)
                cls_dem_week_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif dem_week_s1 <= close < dem_week_p:
                temp = round((close - dem_week_s1) / dem_week_s1, 4)
                cls_dem_week_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - dem_week_p) / dem_week_p, 4)
                cls_dem_week_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif dem_week_p <= close < dem_week_r1:
                temp = round((close - dem_week_p) / dem_week_p, 4)
                cls_dem_week_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
                temp = round((close - dem_week_r1) / dem_week_r1, 4)
                cls_dem_week_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            elif dem_week_r1 <= close:
                temp = round((close - dem_week_r1) / dem_week_r1, 4)
                cls_dem_week_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)

            self.rets.append(
                [
                    self.datas[0].datetime.datetime(), open, high, low, close,
                    self.datas[0].volume[0], self.datas[0].QuoteAssetVolume[0], self.datas[0].NumOfTrades[0], self.datas[0].TakerBuyQuoteAssetVolume[0], self.datas[0].TakerSellQuoteAssetVolume[0],

                    normalizer1to1(round((self.strategy.volInd_5m_12kl.l.sma[0] - self.strategy.volInd_5m_96kl.l.sma[0]) / self.strategy.volInd_5m_96kl.l.sma[0], 4), mono=True, sq=False, division_by=3.0),
                    normalizer1to1(round((self.strategy.volInd_5m_12kl.l.sma[0] * 12 - self.strategy.volInd_1h_8kl.l.sma[0]) / self.strategy.volInd_1h_8kl.l.sma[0], 4), mono=True, sq=False, division_by=3.0),
                    normalizer1to1(round((self.strategy.volInd_5m_12kl.l.sma[0] * 12 - self.strategy.volInd_1h_24kl.l.sma[0]) /self.strategy.volInd_1h_24kl.l.sma[0], 4), mono=True, sq=False, division_by=3.0),
                    normalizer1to1(round((self.strategy.volInd_5m_288kl.l.sma[0] * 12 * 24 - self.strategy.volInd_1d_7kl.l.sma[0]) / self.strategy.volInd_1d_7kl.l.sma[0], 4), mono=True, sq=False, division_by=2.0),
                    0.0,
                    normalizer1to1( round( (self.strategy.volInd_5m_12kl.l.VolPerTrade[0] - self.strategy.volInd_5m_1152kl.l.VolPerTrade[0])/ self.strategy.volInd_5m_1152kl.l.VolPerTrade[0], 4), mono=True, sq=False, division_by=2.0),
                    normalizer1to1( round( (self.strategy.volInd_5m_48kl.l.VolPerTrade[0] - self.strategy.volInd_5m_1152kl.l.VolPerTrade[0])/ self.strategy.volInd_5m_1152kl.l.VolPerTrade[0], 4), mono=True, sq=False, division_by=1.5),
                    normalizer1to1( round( (self.strategy.volInd_5m_96kl.l.VolPerTrade[0] - self.strategy.volInd_5m_1152kl.l.VolPerTrade[0])/ self.strategy.volInd_5m_1152kl.l.VolPerTrade[0], 4), mono=True, sq=False, division_by=1.0),
                    normalizer1to1( round( (self.strategy.volInd_5m_288kl.l.VolPerTrade[0] - self.strategy.volInd_5m_1152kl.l.VolPerTrade[0])/ self.strategy.volInd_5m_1152kl.l.VolPerTrade[0], 4), mono=True, sq=False, division_by=0.5),
                    0.0,
                    normalizer1to1( round( self.strategy.bysl_diff_5m_4kl.l.diff[0], 4), mono=True, sq=False, division_by=3.0),
                    normalizer1to1( round( self.strategy.bysl_diff_5m_12kl.l.diff[0], 4), mono=True, sq=False, division_by=2.5),
                    normalizer1to1( round( self.strategy.bysl_diff_5m_48kl.l.diff[0], 4), mono=True, sq=False, division_by=1.5),
                    normalizer1to1( round( self.strategy.bysl_diff_5m_96kl.l.diff[0], 4), mono=True, sq=False, division_by=1.0),
                    normalizer1to1( round( self.strategy.bysl_diff_5m_288kl.l.diff[0], 4), mono=True, sq=False, division_by=0.5),
                    0.0,
                    round(self.strategy.wR_5m_24kl.l.percR[0], 2),
                    round(self.strategy.wR_5m_48kl.l.percR[0], 2),
                    round(self.strategy.wR_5m_96kl.l.percR[0], 2),
                    round(self.strategy.wR_5m_288kl.l.percR[0], 2),
                    round(self.strategy.wR_5m_1152kl.l.percR[0], 2),
                    0.0,
                    normalizer1to1(self.strategy.Roc_5m_4kl.l.pctchange[0], mono=True, sq=False, division_by=1.0),
                    normalizer1to1(self.strategy.Roc_5m_24kl.l.pctchange[0], mono=True, sq=False, division_by=1.0),
                    normalizer1to1(self.strategy.Roc_5m_48kl.l.pctchange[0], mono=True, sq=False, division_by=1.0),
                    normalizer1to1(self.strategy.Roc_5m_96kl.l.pctchange[0], mono=True, sq=False, division_by=1.0),
                    normalizer1to1(self.strategy.Roc_5m_288kl.l.pctchange[0], mono=True, sq=False, division_by=1.0),
                    normalizer1to1(self.strategy.Roc_5m_1152kl.l.pctchange[0], mono=True, sq=False, division_by=1.0),
                    0.0,
                    cls_keltner_5m_lower25,
                    cls_keltner_5m_mid25,
                    cls_keltner_5m_upper25,
                    0.0,
                    cls_keltner_5m_lower99,
                    cls_keltner_5m_mid99,
                    cls_keltner_5m_upper99,
                    0.0,
                    cls_keltner_1h_lower25,
                    cls_keltner_1h_mid25,
                    cls_keltner_1h_upper25,
                    0.0,
                    cls_keltner_1h_lower99,
                    cls_keltner_1h_mid99,
                    cls_keltner_1h_upper99,
                    0.0,
                    cls_fib_day_s3,
                    cls_fib_day_s2,
                    cls_fib_day_s1,
                    cls_fib_day_p,
                    cls_fib_day_r1,
                    cls_fib_day_r2,
                    cls_fib_day_r3,

                    cls_trad_day_s5,
                    cls_trad_day_s4,
                    cls_trad_day_s3,
                    cls_trad_day_s2,
                    cls_trad_day_s1,
                    cls_trad_day_p,
                    cls_trad_day_r1,
                    cls_trad_day_r2,
                    cls_trad_day_r3,
                    cls_trad_day_r4,
                    cls_trad_day_r5,

                    cls_dem_day_s1,
                    cls_dem_day_p,
                    cls_dem_day_r1,

                    cls_fib_week_s3,
                    cls_fib_week_s2,
                    cls_fib_week_s1,
                    cls_fib_week_p,
                    cls_fib_week_r1,
                    cls_fib_week_r2,
                    cls_fib_week_r3,

                    cls_trad_week_s5,
                    cls_trad_week_s4,
                    cls_trad_week_s3,
                    cls_trad_week_s2,
                    cls_trad_week_s1,
                    cls_trad_week_p,
                    cls_trad_week_r1,
                    cls_trad_week_r2,
                    cls_trad_week_r3,
                    cls_trad_week_r4,
                    cls_trad_week_r5,

                    cls_dem_week_s1,
                    cls_dem_week_p,
                    cls_dem_week_r1,
                ])
        except:
            # traceback.print_exc()
            pass

    def get_analysis(self):
        return self.rets


# Create a Trading Strategy
class VolOsc_wR_RoC_Keltner_PivotPoints(bt.Strategy):
    alias = ('VolOsc_wR_RoC_Keltner_PivotPoints',)
    # list of parameters which are configurable for the strategy
    params = dict(
        pfast=25,  # period for the fast moving average
        pslow=99  # period for the slow moving average
    )

    def log(self, txt, dt=None, tm=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        tm = tm or self.datas[0].datetime.time(0)
        print('%s %s, %s' % (dt.isoformat(), tm.isoformat(), txt))
        # print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Order variable will contain ongoing order details/status
        self.order = None

        self.keltner_5m = KeltnerChannel(self.data0, pfast=25, pslow=99, atr_coe=2)
        self.keltner_1h = KeltnerChannel(self.data1, pfast=25, pslow=99, atr_coe=2)
        
        self.fib_day = bt.ind.FibonacciPivotPoint(self.data2)
        self.trad_day = bt.ind.PivotPoint(self.data2)
        self.dem_day = bt.ind.DemarkPivotPoint(self.data2)

        self.fib_week = bt.ind.FibonacciPivotPoint(self.data3)
        self.trad_week = bt.ind.PivotPoint(self.data3)
        self.dem_week = bt.ind.DemarkPivotPoint(self.data3)

        # self.volMA_5m_12kl = bt.ind.SMA(self.data0.QuoteAssetVolume, period=12)
        # self.volMA_5m_96kl = bt.ind.SMA(self.data0.QuoteAssetVolume, period=96)
        # self.volMA_5m_288kl = bt.ind.SMA(self.data0.QuoteAssetVolume, period=12 * 24)
        # self.volMA_1h_8kl = bt.ind.SMA(self.data1.QuoteAssetVolume, period=8)
        # self.volMA_1h_24kl = bt.ind.SMA(self.data1.QuoteAssetVolume, period=24)
        # self.volMA_1d_7kl = bt.ind.SMA(self.data2.QuoteAssetVolume, period=7)

        self.volInd_5m_12kl = VolumeIndicators(self.data0, period=12)
        self.volInd_5m_48kl = VolumeIndicators(self.data0, period=48)
        self.volInd_5m_96kl = VolumeIndicators(self.data0, period=96)
        self.volInd_5m_288kl = VolumeIndicators(self.data0, period=288)
        self.volInd_5m_1152kl = VolumeIndicators(self.data0, period=1152)
        self.volInd_1h_8kl = VolumeIndicators(self.data1, period=8)
        self.volInd_1h_24kl = VolumeIndicators(self.data1, period=24)
        self.volInd_1d_7kl = VolumeIndicators(self.data2, period=7)

        self.bysl_diff_5m_4kl = BuySellDiff(self.data0, period=4)
        self.bysl_diff_5m_12kl = BuySellDiff(self.data0, period=12)
        self.bysl_diff_5m_48kl = BuySellDiff(self.data0, period=48)
        self.bysl_diff_5m_96kl = BuySellDiff(self.data0, period=96)
        self.bysl_diff_5m_288kl = BuySellDiff(self.data0, period=288)

        # self.rsi_5m_12kl = bt.ind.RSI(self.data0, period=12)
        self.wR_5m_24kl = bt.ind.MY_WilliamsR(self.data0, period=12 * 2)
        self.wR_5m_48kl = bt.ind.MY_WilliamsR(self.data0, period=12 * 4)
        self.wR_5m_96kl = bt.ind.MY_WilliamsR(self.data0, period=12 * 8)
        self.wR_5m_288kl = bt.ind.MY_WilliamsR(self.data0, period=12 * 24)
        self.wR_5m_1152kl = bt.ind.MY_WilliamsR(self.data0, period=12 * 96)

        self.Roc_5m_4kl = bt.ind.PercentChange(self.data0.close, period=4)
        self.Roc_5m_24kl = bt.ind.PercentChange(self.data0.close, period=12 * 2)
        self.Roc_5m_48kl = bt.ind.PercentChange(self.data0.close, period=12 * 4)
        self.Roc_5m_96kl = bt.ind.PercentChange(self.data0.close, period=12 * 8)
        self.Roc_5m_288kl = bt.ind.PercentChange(self.data0.close, period=12 * 24)
        self.Roc_5m_1152kl = bt.ind.PercentChange(self.data0.close, period=12 * 96)

    def next(self):
        pass


class MYGenericCSV(bt.feeds.GenericCSVData):

    # Add a 'pe' line to the inherited ones from the base class
    lines = ('QuoteAssetVolume', 'NumOfTrades', 'TakerBuyQuoteAssetVolume', 'TakerSellQuoteAssetVolume')

    # openinterest in GenericCSVData has index 7 ... add 1
    # add the parameter to the parameters inherited from the base class
    params = (('QuoteAssetVolume', 8), ('NumOfTrades', 9), ('TakerBuyQuoteAssetVolume', 10), ('TakerSellQuoteAssetVolume', 11))


def process_backtrader(c_symbol):
    try:
        cerebro = bt.Cerebro(runonce=False)
        # btdata5m = bt.feeds.GenericCSVData(dataname=c_symbol + '__5M.csv',
        btdata5m = MYGenericCSV(dataname=c_symbol + '__5M.csv',
                                dtformat=('%Y-%m-%d %H:%M:%S'),
                                datetime=0,
                                open=1,
                                high=2,
                                low=3,
                                close=4,
                                volume=5,
                                QuoteAssetVolume=6,
                                NumOfTrades=7,
                                TakerBuyQuoteAssetVolume=8,
                                TakerSellQuoteAssetVolume=9,
                                timeframe=bt.TimeFrame.Minutes,
                                openinterest=-1,
                                # fromdate=date(2021, 2, 16),
                                # todate=date(2021, 2, 16)
                                )

        # Add datafeed to Cerebro Engine
        cerebro.adddata(btdata5m)

        # btdata1h = bt.feeds.GenericCSVData(dataname=c_symbol + '__1H.csv',
        btdata1h = MYGenericCSV(dataname=c_symbol + '__1H.csv',
                                dtformat=('%Y-%m-%d %H:%M:%S'),
                                datetime=0,
                                open=1,
                                high=2,
                                low=3,
                                close=4,
                                volume=5,
                                QuoteAssetVolume=6,
                                NumOfTrades=7,
                                TakerBuyQuoteAssetVolume=8,
                                TakerSellQuoteAssetVolume=9,
                                timeframe=bt.TimeFrame.Minutes,
                                openinterest=-1,
                                # fromdate=date(2021, 2, 16),
                                # todate=date(2021, 2, 16)
                                )

        # Add datafeed to Cerebro Engine
        cerebro.adddata(btdata1h)

        # btdata1d = bt.feeds.GenericCSVData(dataname=c_symbol + '__1D.csv',
        btdata1d = MYGenericCSV(dataname=c_symbol + '__1D.csv',
                                dtformat=('%Y-%m-%d'),
                                datetime=0,
                                open=1,
                                high=2,
                                low=3,
                                close=4,
                                volume=5,
                                QuoteAssetVolume=6,
                                NumOfTrades=7,
                                TakerBuyQuoteAssetVolume=8,
                                TakerSellQuoteAssetVolume=9,
                                timeframe=bt.TimeFrame.Minutes,
                                openinterest=-1,
                                # fromdate=date(2021, 2, 16),
                                # todate=date(2021, 2, 16)
                                )

        # Add datafeed to Cerebro Engine
        cerebro.adddata(btdata1d)

        btdata1w = MYGenericCSV(dataname=c_symbol + '__1W.csv',
                                dtformat=('%Y-%m-%d'),
                                datetime=0,
                                open=1,
                                high=2,
                                low=3,
                                close=4,
                                volume=5,
                                QuoteAssetVolume=6,
                                NumOfTrades=7,
                                TakerBuyQuoteAssetVolume=8,
                                TakerSellQuoteAssetVolume=9,
                                timeframe=bt.TimeFrame.Minutes,
                                openinterest=-1,
                                # fromdate=date(2021, 2, 16),
                                # todate=date(2021, 2, 16)
                                )

        # Add datafeed to Cerebro Engine
        cerebro.adddata(btdata1w)

        # Add Trading Strategy to Cerebro
        cerebro.addstrategy(VolOsc_wR_RoC_Keltner_PivotPoints)
        # Add Trading Statistics Analyzer
        cerebro.addanalyzer(BarAnalysis, _name="bar_data")
        # Run Cerebro Engine
        strat = cerebro.run(runonce=False)
        bar_data_res = strat[0].analyzers.bar_data.get_analysis()
        df = pd.DataFrame(bar_data_res)
        df.columns = ['closeTime', 'open', 'high', 'low', 'close',
                      'volume', 'QuoteAssetVolume', 'NumOfTrades', 'TakerBuyQuoteAssetVolume', 'TakerSellQuoteAssetVolume',
                      
                      'vol5m_osc_12kl-96kl', 'volsum5m12kl-volma1h8kl', 'volsum5m12kl-volma1h24kl', 'volsum5m288kl-volma1d7kl',
                      'zero1',
                      'vpt5mosc_12kl-1152kl', 'vpt5mosc_48kl-1152kl', 'vpt5mosc_96kl-1152kl', 'vpt5mosc_288kl-1152kl',
                      'zero2',
                      'bysldif5m4kl', 'bysldif5m12kl', 'bysldif5m48kl', 'bysldif5m96kl', 'bysldif5m288kl',
                      'zero3',
                      'wR5m24kl', 'wR5m48kl', 'wR5m96kl', 'wR5m288kl', 'wR5m1152kl',
                      'zero4',
                      'Roc5m4kl', 'Roc5m24kl', 'Roc5m48kl', 'Roc5m96kl', 'Roc5m288kl', 'Roc5m1152kl',
                      'zero5',
                      'cls_keltner_5m_lower25', 'cls_keltner_5m_mid25', 'cls_keltner_5m_upper25',
                      'zero6',
                      'cls_keltner_5m_lower99', 'cls_keltner_5m_mid99', 'cls_keltner_5m_upper99',
                      'zero7',
                      'cls_keltner_1h_lower25', 'cls_keltner_1h_mid25', 'cls_keltner_1h_upper25',
                      'zero8',
                      'cls_keltner_1h_lower99', 'cls_keltner_1h_mid99', 'cls_keltner_1h_upper99',
                      'zero9',
                      'cls_fib_day_s3', 'cls_fib_day_s2', 'cls_fib_day_s1', 'cls_fib_day_p', 'cls_fib_day_r1', 'cls_fib_day_r2', 'cls_fib_day_r3',

                      'cls_trad_day_s5', 'cls_trad_day_s4', 'cls_trad_day_s3', 'cls_trad_day_s2', 'cls_trad_day_s1', 'cls_trad_day_p',
                      'cls_trad_day_r1', 'cls_trad_day_r2', 'cls_trad_day_r3', 'cls_trad_day_r4', 'cls_trad_day_r5',

                      'cls_dem_day_s1', 'cls_dem_day_p', 'cls_dem_day_r1',

                      'cls_fib_week_s3', 'cls_fib_week_s2', 'cls_fib_week_s1', 'cls_fib_week_p', 'cls_fib_week_r1', 'cls_fib_week_r2', 'cls_fib_week_r3',

                      'cls_trad_week_s5', 'cls_trad_week_s4', 'cls_trad_week_s3', 'cls_trad_week_s2', 'cls_trad_week_s1', 'cls_trad_week_p',
                      'cls_trad_week_r1', 'cls_trad_week_r2', 'cls_trad_week_r3', 'cls_trad_week_r4', 'cls_trad_week_r5',

                      'cls_dem_week_s1', 'cls_dem_week_p', 'cls_dem_week_r1',]
        
        df.set_index('closeTime', inplace=True)
        # print(df)
        c_df_5m = pd.read_csv(c_symbol + '__5M.csv', index_col=0)
        # print(c_df_5m)
        df2 = df.loc[ :c_df_5m.index[-2] , : ]
        df3 = df.loc[ c_df_5m.index[-1]:c_df_5m.index[-1] , : ]
        df = df2.append(df3.iloc[:1, :])

        # export DataFrame to csv
        df.to_csv(c_symbol + '.csv')
        return c_symbol
    except:
        # printing stack trace
        traceback.print_exc()


def MY_williamsr_last_kl(c_df, period=24):
    h = c_df['high'].iloc[-period:].max()
    l = c_df['low'].iloc[-period:].min()
    c = c_df['close'].iat[-1]
    if h == l:
        pr = 0.5
    else:
        pr = (h - c) / (h - l)
    r = 1.0 - pr
    return r


def process_last_kl_get_featutes(args):
    try:
        c_symbol, kl_1h_pass, last_kl_1h_features, num_ds_row = args
        volMA_1d_7kl, volMA_1h_8kl, volMA_1h_24kl, keltner_1h_lower25, keltner_1h_mid25, keltner_1h_upper25, keltner_1h_lower99, keltner_1h_mid99, keltner_1h_upper99 = last_kl_1h_features
        c_df_5m = pd.read_csv(c_symbol + '__5M.csv', index_col=0)

        c_df_main = pd.read_csv(c_symbol + '.csv', index_col=0)

        c_df_main = c_df_main.append(c_df_5m.iloc[-1:, :])

        volSum_5m_12kl = c_df_5m['QuoteAssetVolume'].tail(12).sum()
        volSum_5m_48kl = c_df_5m['QuoteAssetVolume'].tail(48).sum()
        volSum_5m_96kl = c_df_5m['QuoteAssetVolume'].tail(96).sum()
        volSum_5m_288kl = c_df_5m['QuoteAssetVolume'].tail(288).sum()

        volMA_5m_12kl = volSum_5m_12kl / 12.0
        volMA_5m_12kl = round(volMA_5m_12kl, abs(min(0, int(floor(log10(abs(volMA_5m_12kl))) - 3))))
        volMA_5m_96kl = volSum_5m_96kl / 96.0
        volMA_5m_96kl = round(volMA_5m_96kl, abs(min(0, int(floor(log10(abs(volMA_5m_96kl))) - 3))))
        volMA_5m_288kl = volSum_5m_288kl / 288.0
        volMA_5m_288kl = round(volMA_5m_288kl, abs(min(0, int(floor(log10(abs(volMA_5m_288kl))) - 3))))

        vpt_5m_12kl = 0.0 if volSum_5m_12kl == 0 else (volSum_5m_12kl / c_df_5m['NumOfTrades'].tail(12).sum())
        vpt_5m_48kl = 0.0 if volSum_5m_48kl == 0 else (volSum_5m_48kl / c_df_5m['NumOfTrades'].tail(48).sum())
        vpt_5m_96kl = 0.0 if volSum_5m_96kl == 0 else (volSum_5m_96kl / c_df_5m['NumOfTrades'].tail(96).sum())
        vpt_5m_288kl = 0.0 if volSum_5m_288kl == 0 else (volSum_5m_288kl / c_df_5m['NumOfTrades'].tail(288).sum())
        vpt_5m_1152kl = c_df_5m['QuoteAssetVolume'].tail(1152).sum() / c_df_5m['NumOfTrades'].tail(1152).sum()

        by_5m_4kl = c_df_5m['TakerBuyQuoteAssetVolume'].tail(4).sum()
        sl_5m_4kl = c_df_5m['TakerSellQuoteAssetVolume'].tail(4).sum()
        by_5m_12kl = c_df_5m['TakerBuyQuoteAssetVolume'].tail(12).sum()
        sl_5m_12kl = c_df_5m['TakerSellQuoteAssetVolume'].tail(12).sum()
        by_5m_48kl = c_df_5m['TakerBuyQuoteAssetVolume'].tail(48).sum()
        sl_5m_48kl = c_df_5m['TakerSellQuoteAssetVolume'].tail(48).sum()
        by_5m_96kl = c_df_5m['TakerBuyQuoteAssetVolume'].tail(96).sum()
        sl_5m_96kl = c_df_5m['TakerSellQuoteAssetVolume'].tail(96).sum()
        by_5m_288kl = c_df_5m['TakerBuyQuoteAssetVolume'].tail(288).sum()
        sl_5m_288kl = c_df_5m['TakerSellQuoteAssetVolume'].tail(288).sum()
        bysl_diff_5m_4kl = 100.0 if sl_5m_4kl == 0 else (by_5m_4kl - sl_5m_4kl) / sl_5m_4kl
        bysl_diff_5m_12kl = 100.0 if sl_5m_12kl == 0 else (by_5m_12kl - sl_5m_12kl) / sl_5m_12kl
        bysl_diff_5m_48kl = 100.0 if sl_5m_48kl == 0 else (by_5m_48kl - sl_5m_48kl) / sl_5m_48kl
        bysl_diff_5m_96kl = 100.0 if sl_5m_96kl == 0 else (by_5m_96kl - sl_5m_96kl) / sl_5m_96kl
        bysl_diff_5m_288kl = 100.0 if sl_5m_288kl == 0 else (by_5m_288kl - sl_5m_288kl) / sl_5m_288kl

        wR_5m_24kl = round(MY_williamsr_last_kl(c_df_5m, period=24), 2)
        wR_5m_48kl = round(MY_williamsr_last_kl(c_df_5m, period=48), 2)
        wR_5m_96kl = round(MY_williamsr_last_kl(c_df_5m, period=96), 2)
        wR_5m_288kl = round(MY_williamsr_last_kl(c_df_5m, period=288), 2)
        wR_5m_1152kl = round(MY_williamsr_last_kl(c_df_5m, period=1152), 2)

        Roc_5m_4kl = c_df_5m['close'].iat[-1]/c_df_5m['close'].iat[-4] - 1.0
        Roc_5m_24kl = c_df_5m['close'].iat[-1]/c_df_5m['close'].iat[-24] - 1.0
        Roc_5m_48kl = c_df_5m['close'].iat[-1]/c_df_5m['close'].iat[-48] - 1.0
        Roc_5m_96kl = c_df_5m['close'].iat[-1]/c_df_5m['close'].iat[-96] - 1.0
        Roc_5m_288kl = c_df_5m['close'].iat[-1]/c_df_5m['close'].iat[-288] - 1.0
        Roc_5m_1152kl = c_df_5m['close'].iat[-1]/c_df_5m['close'].iat[-1152] - 1.0

        close_last_5m = c_df_5m['close'].iat[-1]
        
        c_df_5m_26kl = c_df_5m.iloc[-26:, :]
        c_df_5m_100kl = c_df_5m.iloc[-100:, :]
        
        mean = btalib.ema(c_df_5m_26kl['close'], period=25).df.ema.iat[-1]
        atr = btalib.atr(c_df_5m_26kl, _period=1, period=25).df.atr.iat[-1]
        keltner_5m_lower25 = mean - 2 * atr
        keltner_5m_mid25 = mean
        keltner_5m_upper25 = mean + 2 * atr

        keltner_5m_lower25 = round(keltner_5m_lower25, abs(min(0, int(floor(log10(abs(keltner_5m_lower25))) - 3))))
        keltner_5m_mid25 = round(keltner_5m_mid25, abs(min(0, int(floor(log10(abs(keltner_5m_mid25))) - 3))))
        keltner_5m_upper25 = round(keltner_5m_upper25, abs(min(0, int(floor(log10(abs(keltner_5m_upper25))) - 3))))

        mean = btalib.ema(c_df_5m_100kl['close'], period=99).df.ema.iat[-1]
        atr = btalib.atr(c_df_5m_100kl, _period=1, period=99).df.atr.iat[-1]
        keltner_5m_lower99 = mean - 2 * atr
        keltner_5m_mid99 = mean
        keltner_5m_upper99 = mean + 2 * atr

        keltner_5m_lower99 = round(keltner_5m_lower99, abs(min(0, int(floor(log10(abs(keltner_5m_lower99))) - 3))))
        keltner_5m_mid99 = round(keltner_5m_mid99, abs(min(0, int(floor(log10(abs(keltner_5m_mid99))) - 3))))
        keltner_5m_upper99 = round(keltner_5m_upper99, abs(min(0, int(floor(log10(abs(keltner_5m_upper99))) - 3))))

        cls_keltner_5m_lower25 = 0
        cls_keltner_5m_mid25 = 0
        cls_keltner_5m_upper25 = 0
        if close_last_5m < keltner_5m_lower25:
            temp = round((close_last_5m - keltner_5m_lower25) / keltner_5m_lower25, 4)
            cls_keltner_5m_lower25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
        elif keltner_5m_lower25 <= close_last_5m < keltner_5m_mid25:
            temp = round((close_last_5m - keltner_5m_lower25) / keltner_5m_lower25, 4)
            cls_keltner_5m_lower25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
            temp = round((close_last_5m - keltner_5m_mid25) / keltner_5m_mid25, 4)
            cls_keltner_5m_mid25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
        elif keltner_5m_mid25 <= close_last_5m < keltner_5m_upper25:
            temp = round((close_last_5m - keltner_5m_mid25) / keltner_5m_mid25, 4)
            cls_keltner_5m_mid25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
            temp = round((close_last_5m - keltner_5m_upper25) / keltner_5m_upper25, 4)
            cls_keltner_5m_upper25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
        elif keltner_5m_upper25 <= close_last_5m:
            temp = round((close_last_5m - keltner_5m_upper25) / keltner_5m_upper25, 4)
            cls_keltner_5m_upper25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)

        cls_keltner_5m_lower99 = 0
        cls_keltner_5m_mid99 = 0
        cls_keltner_5m_upper99 = 0
        if close_last_5m < keltner_5m_lower99:
            temp = round((close_last_5m - keltner_5m_lower99) / keltner_5m_lower99, 4)
            cls_keltner_5m_lower99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
        elif keltner_5m_lower99 <= close_last_5m < keltner_5m_mid99:
            temp = round((close_last_5m - keltner_5m_lower99) / keltner_5m_lower99, 4)
            cls_keltner_5m_lower99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
            temp = round((close_last_5m - keltner_5m_mid99) / keltner_5m_mid99, 4)
            cls_keltner_5m_mid99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
        elif keltner_5m_mid99 <= close_last_5m < keltner_5m_upper99:
            temp = round((close_last_5m - keltner_5m_mid99) / keltner_5m_mid99, 4)
            cls_keltner_5m_mid99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
            temp = round((close_last_5m - keltner_5m_upper99) / keltner_5m_upper99, 4)
            cls_keltner_5m_upper99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
        elif keltner_5m_upper99 <= close_last_5m:
            temp = round((close_last_5m - keltner_5m_upper99) / keltner_5m_upper99, 4)
            cls_keltner_5m_upper99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)

        if kl_1h_pass or volMA_1d_7kl == 0:
            if volMA_1d_7kl == 0:
                c_df_1d = pd.read_csv(c_symbol + '__1D.csv', index_col=0)
                volMA_1d_7kl = c_df_1d['volume'].tail(7).sum()/7.0
                volMA_1d_7kl = round(volMA_1d_7kl, abs(min(0, int(floor(log10(abs(volMA_1d_7kl))) - 3))))

            c_df_1h = pd.read_csv(c_symbol + '__1H.csv', index_col=0)

            volMA_1h_8kl = c_df_1h['volume'].tail(8).sum()/8.0
            volMA_1h_8kl = round(volMA_1h_8kl, abs(min(0, int(floor(log10(abs(volMA_1h_8kl))) - 3))))
            volMA_1h_24kl = c_df_1h['volume'].tail(24).sum()/24.0
            volMA_1h_24kl = round(volMA_1h_24kl, abs(min(0, int(floor(log10(abs(volMA_1h_24kl))) - 3))))

            c_df_1h_26kl = c_df_1h.iloc[-26:, :]
            c_df_1h_100kl = c_df_1h.iloc[-100:, :]

            mean = btalib.ema(c_df_1h_26kl['close'], period=25).df.ema.iat[-1]
            atr = btalib.atr(c_df_1h_26kl, _period=1, period=25).df.atr.iat[-1]
            keltner_1h_lower25 = mean - 2 * atr
            keltner_1h_mid25 = mean
            keltner_1h_upper25 = mean + 2 * atr

            keltner_1h_lower25 = round(keltner_1h_lower25, abs(min(0, int(floor(log10(abs(keltner_1h_lower25))) - 3))))
            keltner_1h_mid25 = round(keltner_1h_mid25, abs(min(0, int(floor(log10(abs(keltner_1h_mid25))) - 3))))
            keltner_1h_upper25 = round(keltner_1h_upper25, abs(min(0, int(floor(log10(abs(keltner_1h_upper25))) - 3))))

            mean = btalib.ema(c_df_1h_100kl['close'], period=99).df.ema.iat[-1]
            atr = btalib.atr(c_df_1h_100kl, _period=1, period=99).df.atr.iat[-1]
            keltner_1h_lower99 = mean - 2 * atr
            keltner_1h_mid99 = mean
            keltner_1h_upper99 = mean + 2 * atr

            keltner_1h_mid99 = round(keltner_1h_mid99, abs(min(0, int(floor(log10(abs(keltner_1h_mid99))) - 3))))
            keltner_1h_lower99 = round(keltner_1h_lower99, abs(min(0, int(floor(log10(abs(keltner_1h_lower99))) - 3))))
            keltner_1h_upper99 = round(keltner_1h_upper99, abs(min(0, int(floor(log10(abs(keltner_1h_upper99))) - 3))))

        cls_keltner_1h_lower25 = 0
        cls_keltner_1h_mid25 = 0
        cls_keltner_1h_upper25 = 0
        if close_last_5m < keltner_1h_lower25:
            temp = round((close_last_5m - keltner_1h_lower25) / keltner_1h_lower25, 4)
            cls_keltner_1h_lower25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
        elif keltner_1h_lower25 <= close_last_5m < keltner_1h_mid25:
            temp = round((close_last_5m - keltner_1h_lower25) / keltner_1h_lower25, 4)
            cls_keltner_1h_lower25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
            temp = round((close_last_5m - keltner_1h_mid25) / keltner_1h_mid25, 4)
            cls_keltner_1h_mid25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
        elif keltner_1h_mid25 <= close_last_5m < keltner_1h_upper25:
            temp = round((close_last_5m - keltner_1h_mid25) / keltner_1h_mid25, 4)
            cls_keltner_1h_mid25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
            temp = round((close_last_5m - keltner_1h_upper25) / keltner_1h_upper25, 4)
            cls_keltner_1h_upper25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
        elif keltner_1h_upper25 <= close_last_5m:
            temp = round((close_last_5m - keltner_1h_upper25) / keltner_1h_upper25, 4)
            cls_keltner_1h_upper25 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)

        cls_keltner_1h_lower99 = 0
        cls_keltner_1h_mid99 = 0
        cls_keltner_1h_upper99 = 0
        if close_last_5m < keltner_1h_lower99:
            temp = round((close_last_5m - keltner_1h_lower99) / keltner_1h_lower99, 4)
            cls_keltner_1h_lower99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
        elif keltner_1h_lower99 <= close_last_5m < keltner_1h_mid99:
            temp = round((close_last_5m - keltner_1h_lower99) / keltner_1h_lower99, 4)
            cls_keltner_1h_lower99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
            temp = round((close_last_5m - keltner_1h_mid99) / keltner_1h_mid99, 4)
            cls_keltner_1h_mid99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
        elif keltner_1h_mid99 <= close_last_5m < keltner_1h_upper99:
            temp = round((close_last_5m - keltner_1h_mid99) / keltner_1h_mid99, 4)
            cls_keltner_1h_mid99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
            temp = round((close_last_5m - keltner_1h_upper99) / keltner_1h_upper99, 4)
            cls_keltner_1h_upper99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)
        elif keltner_1h_upper99 <= close_last_5m:
            temp = round((close_last_5m - keltner_1h_upper99) / keltner_1h_upper99, 4)
            cls_keltner_1h_upper99 = normalizer1to1(temp, mono=True, sq=False, coef=5.0, division_by=1.0)

        last_index = c_df_main.index[-1]

        c_df_main.at[last_index, 'vol5m_osc_12kl-96kl'] = normalizer1to1( round( (volMA_5m_12kl - volMA_5m_96kl) / volMA_5m_96kl, 4), mono=True, sq=False, division_by=3.0)
        c_df_main.at[last_index, 'volsum5m12kl-volma1h8kl'] = normalizer1to1( round( (volMA_5m_12kl*12 - volMA_1h_8kl) / volMA_1h_8kl, 4), mono=True, sq=False, division_by=3.0)
        c_df_main.at[last_index, 'volsum5m12kl-volma1h24kl'] = normalizer1to1( round( (volMA_5m_12kl*12 - volMA_1h_24kl) / volMA_1h_24kl, 4), mono=True, sq=False, division_by=3.0)
        c_df_main.at[last_index, 'volsum5m288kl-volma1d7kl'] = normalizer1to1(round((volMA_5m_288kl*288 - volMA_1d_7kl) / volMA_1d_7kl, 4), mono=True, sq=False, division_by=2.0)
        c_df_main.at[last_index, 'zero1'] = 0

        c_df_main.at[last_index, 'vpt5mosc_12kl-1152kl'] = normalizer1to1( round( (vpt_5m_12kl - vpt_5m_1152kl)/ vpt_5m_1152kl, 4), mono=True, sq=False, division_by=2.0)
        c_df_main.at[last_index, 'vpt5mosc_48kl-1152kl'] = normalizer1to1( round( (vpt_5m_48kl - vpt_5m_1152kl)/ vpt_5m_1152kl, 4), mono=True, sq=False, division_by=1.5)
        c_df_main.at[last_index, 'vpt5mosc_96kl-1152kl'] = normalizer1to1( round( (vpt_5m_96kl - vpt_5m_1152kl)/ vpt_5m_1152kl, 4), mono=True, sq=False, division_by=1.0)
        c_df_main.at[last_index, 'vpt5mosc_288kl-1152kl'] = normalizer1to1( round( (vpt_5m_288kl - vpt_5m_1152kl)/ vpt_5m_1152kl, 4), mono=True, sq=False, division_by=0.5)
        c_df_main.at[last_index, 'zero2'] = 0

        c_df_main.at[last_index, 'bysldif5m4kl'] = normalizer1to1( round(bysl_diff_5m_4kl, 4), mono=True, sq=False, division_by=2.5)
        c_df_main.at[last_index, 'bysldif5m12kl'] = normalizer1to1( round(bysl_diff_5m_12kl, 4), mono=True, sq=False, division_by=1.5)
        c_df_main.at[last_index, 'bysldif5m48kl'] = normalizer1to1( round(bysl_diff_5m_48kl, 4), mono=True, sq=False, division_by=1.0)
        c_df_main.at[last_index, 'bysldif5m96kl'] = normalizer1to1( round(bysl_diff_5m_96kl, 4), mono=True, sq=False, coef=1.5, division_by=1.0)
        c_df_main.at[last_index, 'bysldif5m288kl'] = normalizer1to1( round(bysl_diff_5m_288kl, 4), mono=True, sq=False, coef=2.5, division_by=1.0)
        c_df_main.at[last_index, 'zero3'] = 0

        c_df_main.at[last_index, 'wR5m24kl'] = wR_5m_24kl
        c_df_main.at[last_index, 'wR5m48kl'] = wR_5m_48kl
        c_df_main.at[last_index, 'wR5m96kl'] = wR_5m_96kl
        c_df_main.at[last_index, 'wR5m288kl'] = wR_5m_288kl
        c_df_main.at[last_index, 'wR5m1152kl'] = wR_5m_1152kl
        c_df_main.at[last_index, 'zero4'] = 0

        c_df_main.at[last_index, 'Roc5m4kl'] = normalizer1to1(Roc_5m_4kl, mono=True, sq=False, coef=5.0, division_by=1.0)
        c_df_main.at[last_index, 'Roc5m24kl'] = normalizer1to1(Roc_5m_24kl, mono=True, sq=False, coef=4.5, division_by=1.0)
        c_df_main.at[last_index, 'Roc5m48kl'] = normalizer1to1(Roc_5m_48kl, mono=True, sq=False, coef=4.0, division_by=1.0)
        c_df_main.at[last_index, 'Roc5m96kl'] = normalizer1to1(Roc_5m_96kl, mono=True, sq=False, coef=3.5, division_by=1.0)
        c_df_main.at[last_index, 'Roc5m288kl'] = normalizer1to1(Roc_5m_288kl, mono=True, sq=False, coef=3.0, division_by=1.0)
        c_df_main.at[last_index, 'Roc5m1152kl'] = normalizer1to1(Roc_5m_1152kl, mono=True, sq=False, coef=2.0, division_by=1.0)
        c_df_main.at[last_index, 'zero5'] = 0
        
        c_df_main.at[last_index,  'cls_keltner_5m_lower25' ] = cls_keltner_5m_lower25
        c_df_main.at[last_index,  'cls_keltner_5m_mid25' ] = cls_keltner_5m_mid25
        c_df_main.at[last_index,  'cls_keltner_5m_upper25' ] = cls_keltner_5m_upper25
        c_df_main.at[last_index, 'zero6'] = 0
        c_df_main.at[last_index, 'cls_keltner_5m_lower99'] = cls_keltner_5m_lower99
        c_df_main.at[last_index, 'cls_keltner_5m_mid99'] = cls_keltner_5m_mid99
        c_df_main.at[last_index, 'cls_keltner_5m_upper99'] = cls_keltner_5m_upper99
        c_df_main.at[last_index, 'zero7'] = 0
        c_df_main.at[last_index, 'cls_keltner_1h_lower25'] = cls_keltner_1h_lower25
        c_df_main.at[last_index, 'cls_keltner_1h_mid25'] = cls_keltner_1h_mid25
        c_df_main.at[last_index, 'cls_keltner_1h_upper25'] = cls_keltner_1h_upper25
        c_df_main.at[last_index, 'zero8'] = 0
        c_df_main.at[last_index, 'cls_keltner_1h_lower99'] = cls_keltner_1h_lower99
        c_df_main.at[last_index, 'cls_keltner_1h_mid99'] = cls_keltner_1h_mid99
        c_df_main.at[last_index, 'cls_keltner_1h_upper99'] = cls_keltner_1h_upper99
        c_df_main.at[last_index, 'zero9'] = 0
        c_df_1d = pd.read_csv(c_symbol + '__1D.csv', index_col=0)
        close_last_1d = c_df_1d['close'].iat[-1]
        open_last_1d = c_df_1d['open'].iat[-1]
        high_last_1d = c_df_1d['high'].iat[-1]
        low_last_1d = c_df_1d['low'].iat[-1]
        level1 = 0.382
        level2 = 0.618
        level3 = 1.000
        fib_day_p = (high_last_1d + low_last_1d + close_last_1d) / 3  # variants duplicate close or add open
        fib_day_s1 = fib_day_p - level1 * (high_last_1d - low_last_1d)
        fib_day_s2 = fib_day_p - level2 * (high_last_1d - low_last_1d)
        fib_day_s3 = fib_day_p - level3 * (high_last_1d - low_last_1d)
        fib_day_r1 = fib_day_p + level1 * (high_last_1d - low_last_1d)
        fib_day_r2 = fib_day_p + level2 * (high_last_1d - low_last_1d)
        fib_day_r3 = fib_day_p + level3 * (high_last_1d - low_last_1d)

        cls_fib_day_s3 = 0
        cls_fib_day_s2 = 0
        cls_fib_day_s1 = 0
        cls_fib_day_p = 0
        cls_fib_day_r1 = 0
        cls_fib_day_r2 = 0
        cls_fib_day_r3 = 0
        if close_last_5m < fib_day_s3:
            temp = round((close_last_5m - fib_day_s3) / fib_day_s3, 4)
            cls_fib_day_s3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif fib_day_s3 <= close_last_5m < fib_day_s2:
            temp = round((close_last_5m - fib_day_s3) / fib_day_s3, 4)
            cls_fib_day_s3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - fib_day_s2) / fib_day_s2, 4)
            cls_fib_day_s2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif fib_day_s2 <= close_last_5m < fib_day_s1:
            temp = round((close_last_5m - fib_day_s2) / fib_day_s2, 4)
            cls_fib_day_s2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - fib_day_s1) / fib_day_s1, 4)
            cls_fib_day_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif fib_day_s1 <= close_last_5m < fib_day_p:
            temp = round((close_last_5m - fib_day_s1) / fib_day_s1, 4)
            cls_fib_day_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - fib_day_p) / fib_day_p, 4)
            cls_fib_day_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif fib_day_p <= close_last_5m < fib_day_r1:
            temp = round((close_last_5m - fib_day_p) / fib_day_p, 4)
            cls_fib_day_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - fib_day_r1) / fib_day_r1, 4)
            cls_fib_day_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif fib_day_r1 <= close_last_5m < fib_day_r2:
            temp = round((close_last_5m - fib_day_r1) / fib_day_r1, 4)
            cls_fib_day_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - fib_day_r2) / fib_day_r2, 4)
            cls_fib_day_r2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif fib_day_r2 <= close_last_5m < fib_day_r3:
            temp = round((close_last_5m - fib_day_r2) / fib_day_r2, 4)
            cls_fib_day_r2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - fib_day_r3) / fib_day_r3, 4)
            cls_fib_day_r3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif fib_day_r3 <= close_last_5m:
            temp = round((close_last_5m - fib_day_r3) / fib_day_r3, 4)
            cls_fib_day_r3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)

        c_df_main.at[last_index, 'cls_fib_day_s3'] = cls_fib_day_s3
        c_df_main.at[last_index, 'cls_fib_day_s2'] = cls_fib_day_s2
        c_df_main.at[last_index, 'cls_fib_day_s1'] = cls_fib_day_s1
        c_df_main.at[last_index, 'cls_fib_day_p'] = cls_fib_day_p
        c_df_main.at[last_index, 'cls_fib_day_r1'] = cls_fib_day_r1
        c_df_main.at[last_index, 'cls_fib_day_r2'] = cls_fib_day_r2
        c_df_main.at[last_index, 'cls_fib_day_r3'] = cls_fib_day_r3


        trad_day_p = (high_last_1d + low_last_1d + close_last_1d) / 3
        trad_day_r1 = trad_day_p * 2 - low_last_1d
        trad_day_s1 = trad_day_p * 2 - high_last_1d
        trad_day_r2 = trad_day_p + (high_last_1d - low_last_1d)
        trad_day_s2 = trad_day_p - (high_last_1d - low_last_1d)
        trad_day_r3 = trad_day_p * 2 + (high_last_1d - 2 * low_last_1d)
        trad_day_s3 = trad_day_p * 2 - (2 * high_last_1d - low_last_1d)
        trad_day_r4 = trad_day_p * 3 + (high_last_1d - 3 * low_last_1d)
        trad_day_s4 = trad_day_p * 3 - (3 * high_last_1d - low_last_1d)
        trad_day_r5 = trad_day_p * 4 + (high_last_1d - 4 * low_last_1d)
        trad_day_s5 = trad_day_p * 4 - (4 * high_last_1d - low_last_1d)

        cls_trad_day_s5 = 0
        cls_trad_day_s4 = 0
        cls_trad_day_s3 = 0
        cls_trad_day_s2 = 0
        cls_trad_day_s1 = 0
        cls_trad_day_p = 0
        cls_trad_day_r1 = 0
        cls_trad_day_r2 = 0
        cls_trad_day_r3 = 0
        cls_trad_day_r4 = 0
        cls_trad_day_r5 = 0
        if close_last_5m < trad_day_s5:
            temp = round((close_last_5m - trad_day_s5) / trad_day_s5, 4)
            cls_trad_day_s5 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_day_s5 <= close_last_5m < trad_day_s4:
            temp = round((close_last_5m - trad_day_s5) / trad_day_s5, 4)
            cls_trad_day_s5 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - trad_day_s4) / trad_day_s4, 4)
            cls_trad_day_s4 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_day_s4 <= close_last_5m < trad_day_s3:
            temp = round((close_last_5m - trad_day_s4) / trad_day_s4, 4)
            cls_trad_day_s4 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - trad_day_s3) / trad_day_s3, 4)
            cls_trad_day_s3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_day_s3 <= close_last_5m < trad_day_s2:
            temp = round((close_last_5m - trad_day_s3) / trad_day_s3, 4)
            cls_trad_day_s3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - trad_day_s2) / trad_day_s2, 4)
            cls_trad_day_s2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_day_s2 <= close_last_5m < trad_day_s1:
            temp = round((close_last_5m - trad_day_s2) / trad_day_s2, 4)
            cls_trad_day_s2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - trad_day_s1) / trad_day_s1, 4)
            cls_trad_day_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_day_s1 <= close_last_5m < trad_day_p:
            temp = round((close_last_5m - trad_day_s1) / trad_day_s1, 4)
            cls_trad_day_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - trad_day_p) / trad_day_p, 4)
            cls_trad_day_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_day_p <= close_last_5m < trad_day_r1:
            temp = round((close_last_5m - trad_day_p) / trad_day_p, 4)
            cls_trad_day_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - trad_day_r1) / trad_day_r1, 4)
            cls_trad_day_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_day_r1 <= close_last_5m < trad_day_r2:
            temp = round((close_last_5m - trad_day_r1) / trad_day_r1, 4)
            cls_trad_day_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - trad_day_r2) / trad_day_r2, 4)
            cls_trad_day_r2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_day_r2 <= close_last_5m < trad_day_r3:
            temp = round((close_last_5m - trad_day_r2) / trad_day_r2, 4)
            cls_trad_day_r2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - trad_day_r3) / trad_day_r3, 4)
            cls_trad_day_r3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_day_r3 <= close_last_5m < trad_day_r4:
            temp = round((close_last_5m - trad_day_r3) / trad_day_r3, 4)
            cls_trad_day_r3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - trad_day_r4) / trad_day_r4, 4)
            cls_trad_day_r4 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_day_r4 <= close_last_5m < trad_day_r5:
            temp = round((close_last_5m - trad_day_r4) / trad_day_r4, 4)
            cls_trad_day_r4 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - trad_day_r5) / trad_day_r5, 4)
            cls_trad_day_r5 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_day_r5 <= close_last_5m:
            temp = round((close_last_5m - trad_day_r5) / trad_day_r5, 4)
            cls_trad_day_r5 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)

        c_df_main.at[last_index, 'cls_trad_day_s5'] = cls_trad_day_s5
        c_df_main.at[last_index, 'cls_trad_day_s4'] = cls_trad_day_s4
        c_df_main.at[last_index, 'cls_trad_day_s3'] = cls_trad_day_s3
        c_df_main.at[last_index, 'cls_trad_day_s2'] = cls_trad_day_s2
        c_df_main.at[last_index, 'cls_trad_day_s1'] = cls_trad_day_s1
        c_df_main.at[last_index, 'cls_trad_day_p'] = cls_trad_day_p
        c_df_main.at[last_index, 'cls_trad_day_r1'] = cls_trad_day_r1
        c_df_main.at[last_index, 'cls_trad_day_r2'] = cls_trad_day_r2
        c_df_main.at[last_index, 'cls_trad_day_r3'] = cls_trad_day_r3
        c_df_main.at[last_index, 'cls_trad_day_r4'] = cls_trad_day_r4
        c_df_main.at[last_index, 'cls_trad_day_r5'] = cls_trad_day_r5

        if close_last_1d < open_last_1d:
            x = high_last_1d + (2*low_last_1d) + close_last_1d
        elif close_last_1d > open_last_1d:
            x = (2*high_last_1d) + low_last_1d + close_last_1d
        elif close_last_1d == open_last_1d:
            x = high_last_1d + low_last_1d + (2*close_last_1d)
        dem_day_p = x / 4
        dem_day_s1 = x / 2 - high_last_1d
        dem_day_r1 = x / 2 - low_last_1d
        cls_dem_day_s1 = 0
        cls_dem_day_p = 0
        cls_dem_day_r1 = 0
        if close_last_5m < dem_day_s1:
            temp = round((close_last_5m - dem_day_s1) / dem_day_s1, 4)
            cls_dem_day_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif dem_day_s1 <= close_last_5m < dem_day_p:
            temp = round((close_last_5m - dem_day_s1) / dem_day_s1, 4)
            cls_dem_day_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - dem_day_p) / dem_day_p, 4)
            cls_dem_day_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif dem_day_p <= close_last_5m < dem_day_r1:
            temp = round((close_last_5m - dem_day_p) / dem_day_p, 4)
            cls_dem_day_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - dem_day_r1) / dem_day_r1, 4)
            cls_dem_day_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif dem_day_r1 <= close_last_5m:
            temp = round((close_last_5m - dem_day_r1) / dem_day_r1, 4)
            cls_dem_day_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)

        c_df_main.at[last_index, 'cls_dem_day_s1'] = cls_dem_day_s1
        c_df_main.at[last_index, 'cls_dem_day_p'] = cls_dem_day_p
        c_df_main.at[last_index, 'cls_dem_day_r1'] = cls_dem_day_r1

        c_df_1w = pd.read_csv(c_symbol + '__1W.csv', index_col=0)
        close_last_1w = c_df_1w['close'].iat[-1]
        open_last_1w = c_df_1w['open'].iat[-1]
        high_last_1w = c_df_1w['high'].iat[-1]
        low_last_1w = c_df_1w['low'].iat[-1]
        level1 = 0.382
        level2 = 0.618
        level3 = 1.000
        fib_week_p = (high_last_1w + low_last_1w + close_last_1w) / 3  # variants duplicate close or add open
        fib_week_s1 = fib_week_p - level1 * (high_last_1w - low_last_1w)
        fib_week_s2 = fib_week_p - level2 * (high_last_1w - low_last_1w)
        fib_week_s3 = fib_week_p - level3 * (high_last_1w - low_last_1w)
        fib_week_r1 = fib_week_p + level1 * (high_last_1w - low_last_1w)
        fib_week_r2 = fib_week_p + level2 * (high_last_1w - low_last_1w)
        fib_week_r3 = fib_week_p + level3 * (high_last_1w - low_last_1w)

        cls_fib_week_s3 = 0
        cls_fib_week_s2 = 0
        cls_fib_week_s1 = 0
        cls_fib_week_p = 0
        cls_fib_week_r1 = 0
        cls_fib_week_r2 = 0
        cls_fib_week_r3 = 0
        if close_last_5m < fib_week_s3:
            temp = round((close_last_5m - fib_week_s3) / fib_week_s3, 4)
            cls_fib_week_s3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif fib_week_s3 <= close_last_5m < fib_week_s2:
            temp = round((close_last_5m - fib_week_s3) / fib_week_s3, 4)
            cls_fib_week_s3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - fib_week_s2) / fib_week_s2, 4)
            cls_fib_week_s2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif fib_week_s2 <= close_last_5m < fib_week_s1:
            temp = round((close_last_5m - fib_week_s2) / fib_week_s2, 4)
            cls_fib_week_s2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - fib_week_s1) / fib_week_s1, 4)
            cls_fib_week_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif fib_week_s1 <= close_last_5m < fib_week_p:
            temp = round((close_last_5m - fib_week_s1) / fib_week_s1, 4)
            cls_fib_week_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - fib_week_p) / fib_week_p, 4)
            cls_fib_week_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif fib_week_p <= close_last_5m < fib_week_r1:
            temp = round((close_last_5m - fib_week_p) / fib_week_p, 4)
            cls_fib_week_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - fib_week_r1) / fib_week_r1, 4)
            cls_fib_week_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif fib_week_r1 <= close_last_5m < fib_week_r2:
            temp = round((close_last_5m - fib_week_r1) / fib_week_r1, 4)
            cls_fib_week_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - fib_week_r2) / fib_week_r2, 4)
            cls_fib_week_r2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif fib_week_r2 <= close_last_5m < fib_week_r3:
            temp = round((close_last_5m - fib_week_r2) / fib_week_r2, 4)
            cls_fib_week_r2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - fib_week_r3) / fib_week_r3, 4)
            cls_fib_week_r3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif fib_week_r3 <= close_last_5m:
            temp = round((close_last_5m - fib_week_r3) / fib_week_r3, 4)
            cls_fib_week_r3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)

        c_df_main.at[last_index, 'cls_fib_week_s3'] = cls_fib_week_s3
        c_df_main.at[last_index, 'cls_fib_week_s2'] = cls_fib_week_s2
        c_df_main.at[last_index, 'cls_fib_week_s1'] = cls_fib_week_s1
        c_df_main.at[last_index, 'cls_fib_week_p'] = cls_fib_week_p
        c_df_main.at[last_index, 'cls_fib_week_r1'] = cls_fib_week_r1
        c_df_main.at[last_index, 'cls_fib_week_r2'] = cls_fib_week_r2
        c_df_main.at[last_index, 'cls_fib_week_r3'] = cls_fib_week_r3


        trad_week_p = (high_last_1w + low_last_1w + close_last_1w) / 3
        trad_week_r1 = trad_week_p * 2 - low_last_1w
        trad_week_s1 = trad_week_p * 2 - high_last_1w
        trad_week_r2 = trad_week_p + (high_last_1w - low_last_1w)
        trad_week_s2 = trad_week_p - (high_last_1w - low_last_1w)
        trad_week_r3 = trad_week_p * 2 + (high_last_1w - 2 * low_last_1w)
        trad_week_s3 = trad_week_p * 2 - (2 * high_last_1w - low_last_1w)
        trad_week_r4 = trad_week_p * 3 + (high_last_1w - 3 * low_last_1w)
        trad_week_s4 = trad_week_p * 3 - (3 * high_last_1w - low_last_1w)
        trad_week_r5 = trad_week_p * 4 + (high_last_1w - 4 * low_last_1w)
        trad_week_s5 = trad_week_p * 4 - (4 * high_last_1w - low_last_1w)

        cls_trad_week_s5 = 0
        cls_trad_week_s4 = 0
        cls_trad_week_s3 = 0
        cls_trad_week_s2 = 0
        cls_trad_week_s1 = 0
        cls_trad_week_p = 0
        cls_trad_week_r1 = 0
        cls_trad_week_r2 = 0
        cls_trad_week_r3 = 0
        cls_trad_week_r4 = 0
        cls_trad_week_r5 = 0
        if close_last_5m < trad_week_s5:
            temp = round((close_last_5m - trad_week_s5) / trad_week_s5, 4)
            cls_trad_week_s5 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_week_s5 <= close_last_5m < trad_week_s4:
            temp = round((close_last_5m - trad_week_s5) / trad_week_s5, 4)
            cls_trad_week_s5 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - trad_week_s4) / trad_week_s4, 4)
            cls_trad_week_s4 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_week_s4 <= close_last_5m < trad_week_s3:
            temp = round((close_last_5m - trad_week_s4) / trad_week_s4, 4)
            cls_trad_week_s4 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - trad_week_s3) / trad_week_s3, 4)
            cls_trad_week_s3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_week_s3 <= close_last_5m < trad_week_s2:
            temp = round((close_last_5m - trad_week_s3) / trad_week_s3, 4)
            cls_trad_week_s3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - trad_week_s2) / trad_week_s2, 4)
            cls_trad_week_s2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_week_s2 <= close_last_5m < trad_week_s1:
            temp = round((close_last_5m - trad_week_s2) / trad_week_s2, 4)
            cls_trad_week_s2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - trad_week_s1) / trad_week_s1, 4)
            cls_trad_week_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_week_s1 <= close_last_5m < trad_week_p:
            temp = round((close_last_5m - trad_week_s1) / trad_week_s1, 4)
            cls_trad_week_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - trad_week_p) / trad_week_p, 4)
            cls_trad_week_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_week_p <= close_last_5m < trad_week_r1:
            temp = round((close_last_5m - trad_week_p) / trad_week_p, 4)
            cls_trad_week_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - trad_week_r1) / trad_week_r1, 4)
            cls_trad_week_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_week_r1 <= close_last_5m < trad_week_r2:
            temp = round((close_last_5m - trad_week_r1) / trad_week_r1, 4)
            cls_trad_week_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - trad_week_r2) / trad_week_r2, 4)
            cls_trad_week_r2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_week_r2 <= close_last_5m < trad_week_r3:
            temp = round((close_last_5m - trad_week_r2) / trad_week_r2, 4)
            cls_trad_week_r2 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - trad_week_r3) / trad_week_r3, 4)
            cls_trad_week_r3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_week_r3 <= close_last_5m < trad_week_r4:
            temp = round((close_last_5m - trad_week_r3) / trad_week_r3, 4)
            cls_trad_week_r3 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - trad_week_r4) / trad_week_r4, 4)
            cls_trad_week_r4 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_week_r4 <= close_last_5m < trad_week_r5:
            temp = round((close_last_5m - trad_week_r4) / trad_week_r4, 4)
            cls_trad_week_r4 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - trad_week_r5) / trad_week_r5, 4)
            cls_trad_week_r5 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif trad_week_r5 <= close_last_5m:
            temp = round((close_last_5m - trad_week_r5) / trad_week_r5, 4)
            cls_trad_week_r5 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)

        c_df_main.at[last_index, 'cls_trad_week_s5'] = cls_trad_week_s5
        c_df_main.at[last_index, 'cls_trad_week_s4'] = cls_trad_week_s4
        c_df_main.at[last_index, 'cls_trad_week_s3'] = cls_trad_week_s3
        c_df_main.at[last_index, 'cls_trad_week_s2'] = cls_trad_week_s2
        c_df_main.at[last_index, 'cls_trad_week_s1'] = cls_trad_week_s1
        c_df_main.at[last_index, 'cls_trad_week_p'] = cls_trad_week_p
        c_df_main.at[last_index, 'cls_trad_week_r1'] = cls_trad_week_r1
        c_df_main.at[last_index, 'cls_trad_week_r2'] = cls_trad_week_r2
        c_df_main.at[last_index, 'cls_trad_week_r3'] = cls_trad_week_r3
        c_df_main.at[last_index, 'cls_trad_week_r4'] = cls_trad_week_r4
        c_df_main.at[last_index, 'cls_trad_week_r5'] = cls_trad_week_r5

        if close_last_1w < open_last_1w:
            x = high_last_1w + (2*low_last_1w) + close_last_1w
        elif close_last_1w > open_last_1w:
            x = (2*high_last_1w) + low_last_1w + close_last_1w
        elif close_last_1w == open_last_1w:
            x = high_last_1w + low_last_1w + (2*close_last_1w)
        dem_week_p = x / 4
        dem_week_s1 = x / 2 - high_last_1w
        dem_week_r1 = x / 2 - low_last_1w
        cls_dem_week_s1 = 0
        cls_dem_week_p = 0
        cls_dem_week_r1 = 0
        if close_last_5m < dem_week_s1:
            temp = round((close_last_5m - dem_week_s1) / dem_week_s1, 4)
            cls_dem_week_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif dem_week_s1 <= close_last_5m < dem_week_p:
            temp = round((close_last_5m - dem_week_s1) / dem_week_s1, 4)
            cls_dem_week_s1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - dem_week_p) / dem_week_p, 4)
            cls_dem_week_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif dem_week_p <= close_last_5m < dem_week_r1:
            temp = round((close_last_5m - dem_week_p) / dem_week_p, 4)
            cls_dem_week_p = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
            temp = round((close_last_5m - dem_week_r1) / dem_week_r1, 4)
            cls_dem_week_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)
        elif dem_week_r1 <= close_last_5m:
            temp = round((close_last_5m - dem_week_r1) / dem_week_r1, 4)
            cls_dem_week_r1 = normalizer1to1(temp, mono=True, sq=False, coef=4.0, division_by=1.0)

        c_df_main.at[last_index, 'cls_dem_week_s1'] = cls_dem_week_s1
        c_df_main.at[last_index, 'cls_dem_week_p'] = cls_dem_week_p
        c_df_main.at[last_index, 'cls_dem_week_r1'] = cls_dem_week_r1

        c_df_main = c_df_main.iloc[ 1: , : ]
        arr_df = c_df_main.iloc[ -1*num_ds_row: , 9: ].to_numpy()
        # ctf1 = arr_df[ -1*num_ds_row: , 64: ]
        # print(c_df_main.iloc[ -1*num_ds_row: , 5: ])
        # print(arr_df)
        # export DataFrame to csv
        c_df_main.to_csv(c_symbol + '.csv')
        return c_symbol, [volMA_1d_7kl, volMA_1h_8kl, volMA_1h_24kl, keltner_1h_lower25, keltner_1h_mid25, keltner_1h_upper25, keltner_1h_lower99, keltner_1h_mid99, keltner_1h_upper99], arr_df

    except:
        # printing stack trace
        traceback.print_exc()

def normalizer1to1(arr_col, mono=True, sq=True, coef=1.0, division_by=2.0, precision=2):
    if mono:
        if sq:
            if arr_col > 0.0:
                normalized = round(tanh(sqrt(coef*arr_col/ division_by)), precision)
            else:
                normalized = -1 * round(tanh(sqrt(-1 * coef*arr_col/ division_by)), precision)
        else:
            normalized = round(tanh(coef*arr_col/division_by), precision)
        return normalized
    else:
        normalized = []
        for i in range(len(arr_col)):
            if sq:
                if arr_col[i] > 0.0:
                    normalized.append(round(tanh(sqrt(coef*arr_col[i] / division_by)), precision))
                else:
                    normalized.append(-1 * round(tanh(sqrt(-1 * coef*arr_col[i] / division_by)), precision))
            else:
                normalized.append(round(tanh(coef*arr_col[i]/division_by), precision))
        return normalized


# def calc_pivot(c_df, fib_T_trad_F=True, day_weak='day'):
#     if fib_T_trad_F:
#         cls_s3 = []
#         cls_s2 = []
#         cls_s1 =[]
#         cls_p = []
#         cls_r1 = []
#         cls_r2 = []
#         cls_r3 = []
#         for i, row in c_df.iterrows():
#             cls_s3.append(0)
#             cls_s2.append(0)
#             cls_s1.append(0)
#             cls_p.append(0)
#             cls_r1.append(0)
#             cls_r2.append(0)
#             cls_r3.append(0)
#             if row['close'] < row['fib_'+day_weak+'_s3']:
#                 temp = round((row['close'] - row['fib_'+day_weak+'_s3']) / row['fib_'+day_weak+'_p'], 4)
#                 cls_s3[-1] = normalizer1to1(temp, mono=True)
#             elif row['fib_'+day_weak+'_s3'] <= row['close'] < row['fib_'+day_weak+'_s2']:
#                 temp = round((row['close'] - row['fib_'+day_weak+'_s3']) / row['fib_'+day_weak+'_p'], 4)
#                 cls_s3[-1] = normalizer1to1(temp, mono=True)
#                 temp = round((row['close'] - row['fib_'+day_weak+'_s2']) / row['fib_'+day_weak+'_p'], 4)
#                 cls_s2[-1] = normalizer1to1(temp, mono=True)
#             elif row['fib_'+day_weak+'_s2'] <= row['close'] < row['fib_'+day_weak+'_s1']:
#                 temp = round((row['close'] - row['fib_'+day_weak+'_s2']) / row['fib_'+day_weak+'_p'], 4)
#                 cls_s2[-1] = normalizer1to1(temp, mono=True)
#                 temp = round((row['close'] - row['fib_'+day_weak+'_s1']) / row['fib_'+day_weak+'_p'], 4)
#                 cls_s1[-1] = normalizer1to1(temp, mono=True)
#             elif row['fib_'+day_weak+'_s1'] <= row['close'] < row['fib_'+day_weak+'_p']:
#                 temp = round((row['close'] - row['fib_'+day_weak+'_s1']) / row['fib_'+day_weak+'_p'], 4)
#                 cls_s1[-1] = normalizer1to1(temp, mono=True)
#                 temp = round((row['close'] - row['fib_'+day_weak+'_p']) / row['fib_'+day_weak+'_p'], 4)
#                 cls_p[-1] = normalizer1to1(temp, mono=True)
#             elif row['fib_'+day_weak+'_p'] <= row['close'] < row['fib_'+day_weak+'_r1']:
#                 temp = round((row['close'] - row['fib_'+day_weak+'_p']) / row['fib_'+day_weak+'_p'], 4)
#                 cls_p[-1] = normalizer1to1(temp, mono=True)
#                 temp = round((row['close'] - row['fib_'+day_weak+'_r1']) / row['fib_'+day_weak+'_p'], 4)
#                 cls_r1[-1] = normalizer1to1(temp, mono=True)
#             elif row['fib_'+day_weak+'_r1'] <= row['close'] < row['fib_'+day_weak+'_r2']:
#                 temp = round((row['close'] - row['fib_'+day_weak+'_r1']) / row['fib_'+day_weak+'_p'], 4)
#                 cls_r1[-1] = normalizer1to1(temp, mono=True)
#                 temp = round((row['close'] - row['fib_'+day_weak+'_r2']) / row['fib_'+day_weak+'_p'], 4)
#                 cls_r2[-1] = normalizer1to1(temp, mono=True)
#             elif row['fib_'+day_weak+'_r2'] <= row['close'] < row['fib_'+day_weak+'_r3']:
#                 temp = round((row['close'] - row['fib_'+day_weak+'_r2']) / row['fib_'+day_weak+'_p'], 4)
#                 cls_r2[-1] = normalizer1to1(temp, mono=True)
#                 temp = round((row['close'] - row['fib_'+day_weak+'_r3']) / row['fib_'+day_weak+'_p'], 4)
#                 cls_r3[-1] = normalizer1to1(temp, mono=True)
#             elif row['fib_'+day_weak+'_r3'] <= row['close']:
#                 temp = round((row['close'] - row['fib_'+day_weak+'_r3']) / row['fib_'+day_weak+'_p'], 4)
#                 cls_r3[-1] = normalizer1to1(temp, mono=True)
#
#         return cls_s3, cls_s2, cls_s1, cls_p, cls_r1, cls_r2, cls_r3
#
#     else:
#         cls_s5 = []
#         cls_s4 = []
#         cls_s3 = []
#         cls_s2 = []
#         cls_s1 = []
#         cls_p = []
#         cls_r1 = []
#         cls_r2 = []
#         cls_r3 = []
#         cls_r4 = []
#         cls_r5 = []
#         for i, row in c_df.iterrows():
#             cls_s5.append(0)
#             cls_s4.append(0)
#             cls_s3.append(0)
#             cls_s2.append(0)
#             cls_s1.append(0)
#             cls_p.append(0)
#             cls_r1.append(0)
#             cls_r2.append(0)
#             cls_r3.append(0)
#             cls_r4.append(0)
#             cls_r5.append(0)
#             if row['close'] < row['trad_'+day_weak+'_s5']:
#                 temp = round((row['close'] - row['trad_'+day_weak+'_s5']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_s5[-1] = normalizer1to1(temp, mono=True)
#             elif row['trad_'+day_weak+'_s5'] <= row['close'] < row['trad_'+day_weak+'_s4']:
#                 temp = round((row['close'] - row['trad_'+day_weak+'_s5']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_s5[-1] = normalizer1to1(temp, mono=True)
#                 temp = round((row['close'] - row['trad_'+day_weak+'_s4']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_s4[-1] = normalizer1to1(temp, mono=True)
#             elif row['trad_'+day_weak+'_s4'] <= row['close'] < row['trad_'+day_weak+'_s3']:
#                 temp = round((row['close'] - row['trad_'+day_weak+'_s4']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_s4[-1] = normalizer1to1(temp, mono=True)
#                 temp = round((row['close'] - row['trad_'+day_weak+'_s3']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_s3[-1] = normalizer1to1(temp, mono=True)
#             elif row['trad_'+day_weak+'_s3'] <= row['close'] < row['trad_'+day_weak+'_s2']:
#                 temp = round((row['close'] - row['trad_'+day_weak+'_s3']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_s3[-1] = normalizer1to1(temp, mono=True)
#                 temp = round((row['close'] - row['trad_'+day_weak+'_s2']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_s2[-1] = normalizer1to1(temp, mono=True)
#             elif row['trad_'+day_weak+'_s2'] <= row['close'] < row['trad_'+day_weak+'_s1']:
#                 temp = round((row['close'] - row['trad_'+day_weak+'_s2']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_s2[-1] = normalizer1to1(temp, mono=True)
#                 temp = round((row['close'] - row['trad_'+day_weak+'_s1']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_s1[-1] = normalizer1to1(temp, mono=True)
#             elif row['trad_'+day_weak+'_s1'] <= row['close'] < row['trad_'+day_weak+'_p']:
#                 temp = round((row['close'] - row['trad_'+day_weak+'_s1']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_s1[-1] = normalizer1to1(temp, mono=True)
#                 temp = round((row['close'] - row['trad_'+day_weak+'_p']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_p[-1] = normalizer1to1(temp, mono=True)
#             elif row['trad_'+day_weak+'_p'] <= row['close'] < row['trad_'+day_weak+'_r1']:
#                 temp = round((row['close'] - row['trad_'+day_weak+'_p']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_p[-1] = normalizer1to1(temp, mono=True)
#                 temp = round((row['close'] - row['trad_'+day_weak+'_r1']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_r1[-1] = normalizer1to1(temp, mono=True)
#             elif row['trad_'+day_weak+'_r1'] <= row['close'] < row['trad_'+day_weak+'_r2']:
#                 temp = round((row['close'] - row['trad_'+day_weak+'_r1']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_r1[-1] = normalizer1to1(temp, mono=True)
#                 temp = round((row['close'] - row['trad_'+day_weak+'_r2']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_r2[-1] = normalizer1to1(temp, mono=True)
#             elif row['trad_'+day_weak+'_r2'] <= row['close'] < row['trad_'+day_weak+'_r3']:
#                 temp = round((row['close'] - row['trad_'+day_weak+'_r2']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_r2[-1] = normalizer1to1(temp, mono=True)
#                 temp = round((row['close'] - row['trad_'+day_weak+'_r3']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_r3[-1] = normalizer1to1(temp, mono=True)
#             elif row['trad_'+day_weak+'_r3'] <= row['close'] < row['trad_'+day_weak+'_r4']:
#                 temp = round((row['close'] - row['trad_'+day_weak+'_r3']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_r3[-1] = normalizer1to1(temp, mono=True)
#                 temp = round((row['close'] - row['trad_'+day_weak+'_r4']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_r4[-1] = normalizer1to1(temp, mono=True)
#             elif row['trad_'+day_weak+'_r4'] <= row['close'] < row['trad_'+day_weak+'_r5']:
#                 temp = round((row['close'] - row['trad_'+day_weak+'_r4']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_r4[-1] = normalizer1to1(temp, mono=True)
#                 temp = round((row['close'] - row['trad_'+day_weak+'_r5']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_r5[-1] = normalizer1to1(temp, mono=True)
#             elif row['trad_'+day_weak+'_r5'] <= row['close']:
#                 temp = round((row['close'] - row['trad_'+day_weak+'_r5']) / row['trad_'+day_weak+'_p'], 4)
#                 cls_r5[-1] = normalizer1to1(temp, mono=True)
#
#         return cls_s5, cls_s4, cls_s3, cls_s2, cls_s1, cls_p, cls_r1, cls_r2, cls_r3, cls_r4, cls_r5
#
#
# def calc_keltner(c_df, _5m_1h='5m', _25kl_99kl='25'):
#     cls_keltner_lower = []
#     cls_keltner_mid = []
#     cls_keltner_upper =[]
#     for i, row in c_df.iterrows():
#         cls_keltner_lower.append(0)
#         cls_keltner_mid.append(0)
#         cls_keltner_upper.append(0)
#         if row['close'] < row['keltner_'+_5m_1h+'_lower'+_25kl_99kl]:
#             temp = round((row['close'] - row['keltner_'+_5m_1h+'_lower'+_25kl_99kl]) / row['keltner_'+_5m_1h+'_mid'+_25kl_99kl], 4)
#             cls_keltner_lower[-1] = normalizer1to1(temp, mono=True)
#         elif row['keltner_'+_5m_1h+'_lower'+_25kl_99kl] <= row['close'] < row['keltner_'+_5m_1h+'_mid'+_25kl_99kl]:
#             temp = round((row['close'] - row['keltner_'+_5m_1h+'_lower'+_25kl_99kl]) / row['keltner_'+_5m_1h+'_mid'+_25kl_99kl], 4)
#             cls_keltner_lower[-1] = normalizer1to1(temp, mono=True)
#             temp = round((row['close'] - row['keltner_'+_5m_1h+'_mid'+_25kl_99kl]) / row['keltner_'+_5m_1h+'_mid'+_25kl_99kl], 4)
#             cls_keltner_mid[-1] = normalizer1to1(temp, mono=True)
#         elif row['keltner_'+_5m_1h+'_mid'+_25kl_99kl] <= row['close'] < row['keltner_'+_5m_1h+'_upper'+_25kl_99kl]:
#             temp = round((row['close'] - row['keltner_'+_5m_1h+'_mid'+_25kl_99kl]) / row['keltner_'+_5m_1h+'_mid'+_25kl_99kl], 4)
#             cls_keltner_mid[-1] = normalizer1to1(temp, mono=True)
#             temp = round((row['close'] - row['keltner_'+_5m_1h+'_upper'+_25kl_99kl]) / row['keltner_'+_5m_1h+'_mid'+_25kl_99kl], 4)
#             cls_keltner_upper[-1] = normalizer1to1(temp, mono=True)
#         elif row['keltner_'+_5m_1h+'_upper'+_25kl_99kl] <= row['close']:
#             temp = round((row['close'] - row['keltner_'+_5m_1h+'_upper'+_25kl_99kl]) / row['keltner_'+_5m_1h+'_mid'+_25kl_99kl], 4)
#             cls_keltner_upper[-1] = normalizer1to1(temp, mono=True)
#
#     return cls_keltner_lower, cls_keltner_mid, cls_keltner_upper
#
#
# def process_get_features(c_symbol):
#     try:
#         # load DataFrame
#         c_df = pd.read_csv(c_symbol + '.csv', index_col=0)
#
#         c_df['cls_keltner_5m_lower25'], c_df['cls_keltner_5m_mid25'], c_df['cls_keltner_5m_upper25'] = calc_keltner(c_df, _5m_1h='5m', _25kl_99kl='25')
#
#         c_df['cls_keltner_5m_lower99'], c_df['cls_keltner_5m_mid99'], c_df['cls_keltner_5m_upper99'] = calc_keltner(c_df, _5m_1h='5m', _25kl_99kl='99')
#
#         c_df['cls_keltner_1h_lower25'], c_df['cls_keltner_1h_mid25'], c_df['cls_keltner_1h_upper25'] = calc_keltner(c_df, _5m_1h='1h', _25kl_99kl='25')
#
#         c_df['cls_keltner_1h_lower99'], c_df['cls_keltner_1h_mid99'], c_df['cls_keltner_1h_upper99'] = calc_keltner(c_df, _5m_1h='1h', _25kl_99kl='99')
#
#         c_df['cls_fib_day_s3'] = np.nan
#         c_df['cls_fib_day_s2'] = np.nan
#         c_df['cls_fib_day_s1'] = np.nan
#         c_df['cls_fib_day_p'] = np.nan
#         c_df['cls_fib_day_r1'] = np.nan
#         c_df['cls_fib_day_r2'] = np.nan
#         c_df['cls_fib_day_r3'] = np.nan
#         c_df['cls_fib_day_s3'], c_df['cls_fib_day_s2'], c_df['cls_fib_day_s1'], c_df['cls_fib_day_p'], \
#         c_df['cls_fib_day_r1'], c_df['cls_fib_day_r2'], c_df['cls_fib_day_r3'] = calc_pivot(c_df, fib_T_trad_F=True, day_weak='day')
#
#         c_df['cls_trad_day_s5'] = np.nan
#         c_df['cls_trad_day_s4'] = np.nan
#         c_df['cls_trad_day_s3'] = np.nan
#         c_df['cls_trad_day_s2'] = np.nan
#         c_df['cls_trad_day_s1'] = np.nan
#         c_df['cls_trad_day_p'] = np.nan
#         c_df['cls_trad_day_r1'] = np.nan
#         c_df['cls_trad_day_r2'] = np.nan
#         c_df['cls_trad_day_r3'] = np.nan
#         c_df['cls_trad_day_r4'] = np.nan
#         c_df['cls_trad_day_r5'] = np.nan
#         c_df['cls_trad_day_s5'], c_df['cls_trad_day_s4'], c_df['cls_trad_day_s3'], c_df['cls_trad_day_s2'], \
#         c_df['cls_trad_day_s1'], c_df['cls_trad_day_p'], c_df['cls_trad_day_r1'], c_df['cls_trad_day_r2'], \
#         c_df['cls_trad_day_r3'], c_df['cls_trad_day_r4'], c_df['cls_trad_day_r5'] = calc_pivot(c_df, fib_T_trad_F=False, day_weak='day')
#
#         c_df['cls_fib_week_s3'] = np.nan
#         c_df['cls_fib_week_s2'] = np.nan
#         c_df['cls_fib_week_s1'] = np.nan
#         c_df['cls_fib_week_p'] = np.nan
#         c_df['cls_fib_week_r1'] = np.nan
#         c_df['cls_fib_week_r2'] = np.nan
#         c_df['cls_fib_week_r3'] = np.nan
#         c_df['cls_fib_week_s3'], c_df['cls_fib_week_s2'], c_df['cls_fib_week_s1'], c_df['cls_fib_week_p'], \
#         c_df['cls_fib_week_r1'], c_df['cls_fib_week_r2'], c_df['cls_fib_week_r3'] = calc_pivot(c_df, fib_T_trad_F=True, day_weak='week')
#
#         c_df['cls_trad_week_s5'] = np.nan
#         c_df['cls_trad_week_s4'] = np.nan
#         c_df['cls_trad_week_s3'] = np.nan
#         c_df['cls_trad_week_s2'] = np.nan
#         c_df['cls_trad_week_s1'] = np.nan
#         c_df['cls_trad_week_p'] = np.nan
#         c_df['cls_trad_week_r1'] = np.nan
#         c_df['cls_trad_week_r2'] = np.nan
#         c_df['cls_trad_week_r3'] = np.nan
#         c_df['cls_trad_week_r4'] = np.nan
#         c_df['cls_trad_week_r5'] = np.nan
#         c_df['cls_trad_week_s5'], c_df['cls_trad_week_s4'], c_df['cls_trad_week_s3'], c_df['cls_trad_week_s2'], \
#         c_df['cls_trad_week_s1'], c_df['cls_trad_week_p'], c_df['cls_trad_week_r1'], c_df['cls_trad_week_r2'], \
#         c_df['cls_trad_week_r3'], c_df['cls_trad_week_r4'], c_df['cls_trad_week_r5'] = calc_pivot(c_df, fib_T_trad_F=False, day_weak='week')
#         # export DataFrame to csv
#         c_df.to_csv(c_symbol + '.csv')
#         return c_symbol
#
#     except:
#         # printing stack trace
#         traceback.print_exc()


def _1w_history_receiver( c_symbol ) :
    api_key_history = 'yours'
    api_secret_history = 'yours'
    connected = False
    attempts = 5
    sleep(round(random.uniform(0.0, 7.0), 2))
    while attempts > 0:
        try:
            b_client = Client( api_key_history, api_secret_history )
            connected = True
            break
        except:
            # # printing stack trace
            # traceback.print_exc()
            attempts -= 1
            sleep(round(random.uniform(0.5, 2.0), 2))

    if connected:
        interval = b_client.KLINE_INTERVAL_1WEEK
        start = '15 days ago UTC'  # '9 hours ago UTC', '12 hours ago UTC', '1 day ago UTC' , '1 Jan, 2021', '15 Dec, 2020' ,
        # end = default: None , e.g.: "1 Jan, 2021"
        nb_klines = 2

        received = False
        while not received:
            try:
                # Fetch klines for any date range and interval
                klines = b_client.get_historical_klines(c_symbol, interval, start)  # ,end)
                received = True
                # [
                #     1609756500000,      # Open time
                #     '29412.53000000',   # Open
                #     '29687.55000000',   # High
                #     '29387.61000000',   # Low
                #     '29424.99000000',   # Close
                #     '736.49694300',     # Volume
                #     1609756799999,      # Close time
                #     '21753979.14647961',# Quote asset volume
                #     9433,               # Number of trades
                #     '418.77505800',     # Taker buy base asset volume
                #     '12364403.68504647',# Taker buy quote asset volume
                #     '0'                 # Ignore
                #  ]
            except:
                sleep(round(random.uniform(0.5, 2.0), 2))

        try:
            if len(klines) < nb_klines:
                print(c_symbol+': low kline')
                return c_symbol, 'ZZZ'
            else:
                for line in klines:
                    del line[11]
                    # del line[10]
                    del line[9]
                    # del line[8]
                    # del line[7]
                    del line[6]
                    # del line[1]

                # option 4 - create a Pandas DataFrame and export to CSV
                c_df_1w = pd.DataFrame(klines, columns=['openTime', 'open', 'high', 'low', 'close', 'volume',
                                                        'QuoteAssetVolume', 'NumOfTrades', 'TakerBuyQuoteAssetVolume'])
                # c_df_1w = pd.DataFrame(klines, columns=['openTime', 'open', 'high', 'low', 'close', 'volume'])
                c_df_1w.set_index( 'openTime', inplace = True )
                c_df_1w.index = pd.to_datetime( c_df_1w.index, unit = 'ms' )
                c_df_1w['closeTime'] = pd.to_datetime(c_df_1w.index.values) + timedelta(weeks=1)
                c_df_1w.set_index('closeTime', inplace=True)
                c_df_1w['open'] = pd.to_numeric( c_df_1w.open, downcast = 'float' )
                c_df_1w['high'] = pd.to_numeric( c_df_1w.high, downcast = 'float' )
                c_df_1w['low'] = pd.to_numeric( c_df_1w.low, downcast = 'float' )
                c_df_1w['close'] = pd.to_numeric( c_df_1w.close, downcast = 'float' )
                c_df_1w['volume'] = pd.to_numeric( c_df_1w.volume, downcast = 'float' )
                c_df_1w['NumOfTrades'] = pd.to_numeric(c_df_1w.NumOfTrades, downcast='integer')
                c_df_1w['QuoteAssetVolume'] = pd.to_numeric(c_df_1w.QuoteAssetVolume, downcast='float')
                c_df_1w['TakerBuyQuoteAssetVolume'] = pd.to_numeric(c_df_1w.TakerBuyQuoteAssetVolume, downcast='float')
                c_df_1w['TakerSellQuoteAssetVolume'] = c_df_1w['QuoteAssetVolume'] - c_df_1w['TakerBuyQuoteAssetVolume']

                c_df_1w = c_df_1w.iloc[ :-1 , : ]
                c_df_1w.to_csv(c_symbol + '__1W.csv')
                return c_symbol, klines[0][0]
        except:
            # printing stack trace
            traceback.print_exc()
    else:
        print(c_symbol+': not connect')
        return c_symbol, 'ZZZ'


def _1d_history_receiver( args ) :
    api_key_history = 'yours'
    api_secret_history = 'yours'
    connected = False
    attempts = 5
    sleep(round(random.uniform(0.0, 1.0), 2))
    while attempts > 0:
        try:
            b_client = Client( api_key_history, api_secret_history )
            connected = True
            break
        except:
            # # printing stack trace
            # traceback.print_exc()
            attempts -= 1
            sleep(round(random.uniform(0.5, 1.0), 2))

    c_symbol, open_time_w = args
    if connected:
        interval = b_client.KLINE_INTERVAL_1DAY
        start = '7 days ago UTC'  # '9 hours ago UTC', '12 hours ago UTC', '1 day ago UTC' , '1 Jan, 2021', '15 Dec, 2020' ,
        # end = default: None , e.g.: "1 Jan, 2021"
        nb_klines = 7

        received = False
        while not received:
            try:
                # Fetch klines for any date range and interval
                klines = b_client.get_historical_klines(c_symbol, interval, start)  # ,end)
                received = True
                # [
                #     1609756500000,      # Open time
                #     '29412.53000000',   # Open
                #     '29687.55000000',   # High
                #     '29387.61000000',   # Low
                #     '29424.99000000',   # Close
                #     '736.49694300',     # Volume
                #     1609756799999,      # Close time
                #     '21753979.14647961',# Quote asset volume
                #     9433,               # Number of trades
                #     '418.77505800',     # Taker buy base asset volume
                #     '12364403.68504647',# Taker buy quote asset volume
                #     '0'                 # Ignore
                #  ]
            except:
                sleep(round(random.uniform(0.5, 1.0), 2))

        try:
            if len(klines) < nb_klines:
                print(c_symbol+': low kline')
                return 'ZZZ'
            else:
                for line in klines:
                    del line[11]
                    # del line[10]
                    del line[9]
                    # del line[8]
                    # del line[7]
                    del line[6]
                    # del line[1]

                # option 4 - create a Pandas DataFrame and export to CSV
                c_df_1d = pd.DataFrame(klines, columns=['openTime', 'open', 'high', 'low', 'close', 'volume',
                                                        'QuoteAssetVolume', 'NumOfTrades', 'TakerBuyQuoteAssetVolume'])
                # c_df_1d = pd.DataFrame(klines, columns=['openTime', 'open', 'high', 'low', 'close', 'volume'])
                open_time_1d = klines[0][0]
                c_df_1d_beg = c_df_1d_app = c_df_1d.iloc[0:1, :]
                for _ in range(open_time_w, open_time_1d-1000*60*60*24, 1000*60*60*24):
                    c_df_1d_beg = c_df_1d_beg.append(c_df_1d_app)
                c_df_1d_beg['openTime'] = range(open_time_w, open_time_1d, 1000*60*60*24)
                c_df_1d = pd.concat([c_df_1d_beg, c_df_1d]).reset_index(drop=True)

                c_df_1d.set_index( 'openTime', inplace = True )
                c_df_1d.index = pd.to_datetime( c_df_1d.index, unit = 'ms' )
                c_df_1d['closeTime'] = pd.to_datetime(c_df_1d.index.values) + timedelta(days=1)
                c_df_1d.set_index('closeTime', inplace=True)
                c_df_1d['open'] = pd.to_numeric( c_df_1d.open, downcast = 'float' )
                c_df_1d['high'] = pd.to_numeric( c_df_1d.high, downcast = 'float' )
                c_df_1d['low'] = pd.to_numeric( c_df_1d.low, downcast = 'float' )
                c_df_1d['close'] = pd.to_numeric( c_df_1d.close, downcast = 'float' )
                c_df_1d['volume'] = pd.to_numeric( c_df_1d.volume, downcast = 'float' )
                c_df_1d['NumOfTrades'] = pd.to_numeric(c_df_1d.NumOfTrades, downcast='integer')

                c_df_1d['QuoteAssetVolume'] = pd.to_numeric(c_df_1d.QuoteAssetVolume, downcast='float')
                c_df_1d['TakerBuyQuoteAssetVolume'] = pd.to_numeric(c_df_1d.TakerBuyQuoteAssetVolume, downcast='float')
                c_df_1d['TakerSellQuoteAssetVolume'] = c_df_1d['QuoteAssetVolume'] - c_df_1d['TakerBuyQuoteAssetVolume']

                c_df_1d = c_df_1d.iloc[:-1, :]

                c_df_1d.to_csv(c_symbol + '__1D.csv')
                return c_symbol
        except:
            # printing stack trace
            traceback.print_exc()
    else:
        print(c_symbol+': not connect')
        return 'ZZZ'


def _1h_history_receiver( args ) :
    api_key_history = 'yours'
    api_secret_history = 'yours'
    connected = False
    attempts = 5
    sleep(round(random.uniform(0.0, 1.0), 2))
    while attempts > 0:
        try:
            b_client = Client( api_key_history, api_secret_history )
            connected = True
            break
        except:
            # # printing stack trace
            # traceback.print_exc()
            attempts -= 1
            sleep(round(random.uniform(0.5, 1.0), 2))

    c_symbol, open_time_w = args
    if connected:
        interval = b_client.KLINE_INTERVAL_1HOUR
        start = '185 hours ago UTC'  # '9 hours ago UTC', '12 hours ago UTC', '1 day ago UTC' , '1 Jan, 2021', '15 Dec, 2020' ,
        # end = default: None , e.g.: "1 Jan, 2021"
        nb_klines = 184

        received = False
        while not received:
            try:
                # with binanceConnectionLock :
                # Fetch klines for any date range and interval
                klines = b_client.get_historical_klines(c_symbol, interval, start)  # ,end)
                received = True
                # [
                #     1609756500000,      # Open time
                #     '29412.53000000',   # Open
                #     '29687.55000000',   # High
                #     '29387.61000000',   # Low
                #     '29424.99000000',   # Close
                #     '736.49694300',     # Volume
                #     1609756799999,      # Close time
                #     '21753979.14647961',# Quote asset volume
                #     9433,               # Number of trades
                #     '418.77505800',     # Taker buy base asset volume
                #     '12364403.68504647',# Taker buy quote asset volume
                #     '0'                 # Ignore
                #  ]
            except:
                sleep(round(random.uniform(0.5, 1.0), 2))

        try:
            if len(klines) < nb_klines:
                print(c_symbol+': low kline')
                return 'ZZZ'
            else:
                for line in klines:
                    del line[11]
                    # del line[10]
                    del line[9]
                    # del line[8]
                    # del line[7]
                    del line[6]
                    # del line[1]

                # option 4 - create a Pandas DataFrame and export to CSV
                c_df_1h = pd.DataFrame(klines, columns=['openTime', 'open', 'high', 'low', 'close', 'volume',
                                                        'QuoteAssetVolume', 'NumOfTrades', 'TakerBuyQuoteAssetVolume'])
                # c_df_1h = pd.DataFrame(klines, columns=['openTime', 'open', 'high', 'low', 'close', 'volume'])
                open_time_1h = klines[0][0]
                c_df_1h_beg = c_df_1h_app = c_df_1h.iloc[0:1, :]
                for _ in range(open_time_w, open_time_1h - 1000*60*60, 1000*60*60):
                    c_df_1h_beg = c_df_1h_beg.append(c_df_1h_app)
                c_df_1h_beg['openTime'] = range(open_time_w, open_time_1h, 1000*60*60)
                c_df_1h = pd.concat([c_df_1h_beg, c_df_1h]).reset_index(drop=True)

                c_df_1h.set_index( 'openTime', inplace = True )
                c_df_1h.index = pd.to_datetime( c_df_1h.index, unit = 'ms' )
                c_df_1h['closeTime'] = pd.to_datetime(c_df_1h.index.values) + timedelta(hours=1)
                c_df_1h.set_index('closeTime', inplace=True)
                c_df_1h['open'] = pd.to_numeric( c_df_1h.open, downcast = 'float' )
                c_df_1h['high'] = pd.to_numeric( c_df_1h.high, downcast = 'float' )
                c_df_1h['low'] = pd.to_numeric( c_df_1h.low, downcast = 'float' )
                c_df_1h['close'] = pd.to_numeric( c_df_1h.close, downcast = 'float' )
                c_df_1h['volume'] = pd.to_numeric( c_df_1h.volume, downcast = 'float' )
                c_df_1h['NumOfTrades'] = pd.to_numeric(c_df_1h.NumOfTrades, downcast='integer')

                c_df_1h['QuoteAssetVolume'] = pd.to_numeric(c_df_1h.QuoteAssetVolume, downcast='float')
                c_df_1h['TakerBuyQuoteAssetVolume'] = pd.to_numeric(c_df_1h.TakerBuyQuoteAssetVolume, downcast='float')
                c_df_1h['TakerSellQuoteAssetVolume'] = c_df_1h['QuoteAssetVolume'] - c_df_1h['TakerBuyQuoteAssetVolume']

                # export DataFrame to csv
                c_df_1h.to_csv(c_symbol + '__1H.csv')

                return c_symbol

        except:
            # printing stack trace
            traceback.print_exc()
    else:
        print(c_symbol+': not connect')
        return 'ZZZ'


def _5m_history_receiver( args ) :
    api_key_history = 'yours'
    api_secret_history = 'yours'
    connected = False
    attempts = 5
    sleep(round(random.uniform(0.0, 1.0), 2))
    while attempts > 0:
        try:
            b_client = Client( api_key_history, api_secret_history )
            connected = True
            break
        except:
            # # printing stack trace
            # traceback.print_exc()
            attempts -= 1
            sleep(round(random.uniform(0.5, 1.0), 2))

    c_symbol, open_time_w = args
    if connected:
        interval = b_client.KLINE_INTERVAL_5MINUTE
        start = '104 hours ago UTC'  # '9 hours ago UTC', '12 hours ago UTC', '1 day ago UTC' , '1 Jan, 2021', '15 Dec, 2020' ,
        # end = default: None , e.g.: "1 Jan, 2021"
        nb_klines = 1236

        received = False
        while not received:
            try:
                # with binanceConnectionLock :
                # Fetch klines for any date range and interval
                klines = b_client.get_historical_klines(c_symbol, interval, start)  # ,end)
                received = True
                # [
                #     1609756500000,      # Open time
                #     '29412.53000000',   # Open
                #     '29687.55000000',   # High
                #     '29387.61000000',   # Low
                #     '29424.99000000',   # Close
                #     '736.49694300',     # Volume
                #     1609756799999,      # Close time
                #     '21753979.14647961',# Quote asset volume
                #     9433,               # Number of trades
                #     '418.77505800',     # Taker buy base asset volume
                #     '12364403.68504647',# Taker buy quote asset volume
                #     '0'                 # Ignore
                #  ]
            except:
                sleep(round(random.uniform(0.5, 1.0), 2))

        try:
            if len(klines) < nb_klines:
                print(c_symbol+': low kline: '+str(len(klines)))
                return 'ZZZ'
            else:
                for line in klines:
                    del line[11]
                    # del line[10]
                    del line[9]
                    # del line[8]
                    # del line[7]
                    del line[6]
                    # del line[1]

                # option 4 - create a Pandas DataFrame and export to CSV
                c_df_5m = pd.DataFrame(klines, columns=['openTime', 'open', 'high', 'low', 'close', 'volume',
                                                        'QuoteAssetVolume', 'NumOfTrades', 'TakerBuyQuoteAssetVolume'])
                # c_df_5m = pd.DataFrame(klines, columns=['openTime', 'open', 'high', 'low', 'close', 'volume'])
                open_time_5m = klines[0][0]
                c_df_5m_beg = c_df_5m_app = c_df_5m.iloc[0:1, :]
                for _ in range(open_time_w, open_time_5m - 1000 * 60 * 5, 1000 * 60 * 5):
                    c_df_5m_beg = c_df_5m_beg.append(c_df_5m_app)
                c_df_5m_beg['openTime'] = range(open_time_w, open_time_5m, 1000 * 60 * 5)
                c_df_5m = pd.concat([c_df_5m_beg, c_df_5m]).reset_index(drop=True)

                c_df_5m.set_index( 'openTime', inplace = True )
                c_df_5m.index = pd.to_datetime( c_df_5m.index, unit = 'ms' )
                c_df_5m['closeTime'] = pd.to_datetime(c_df_5m.index.values) + timedelta(minutes=5)
                c_df_5m.set_index('closeTime', inplace=True)
                c_df_5m['open'] = pd.to_numeric( c_df_5m.open, downcast = 'float' )
                c_df_5m['high'] = pd.to_numeric( c_df_5m.high, downcast = 'float' )
                c_df_5m['low'] = pd.to_numeric( c_df_5m.low, downcast = 'float' )
                c_df_5m['close'] = pd.to_numeric( c_df_5m.close, downcast = 'float' )
                c_df_5m['volume'] = pd.to_numeric( c_df_5m.volume, downcast = 'float' )
                c_df_5m['NumOfTrades'] = pd.to_numeric(c_df_5m.NumOfTrades, downcast='integer')

                c_df_5m['QuoteAssetVolume'] = pd.to_numeric(c_df_5m.QuoteAssetVolume, downcast='float')
                c_df_5m['TakerBuyQuoteAssetVolume'] = pd.to_numeric(c_df_5m.TakerBuyQuoteAssetVolume, downcast='float')
                c_df_5m['TakerSellQuoteAssetVolume'] = c_df_5m['QuoteAssetVolume'] - c_df_5m['TakerBuyQuoteAssetVolume']

                # export DataFrame to csv
                c_df_5m.to_csv(c_symbol + '__5M.csv')
                # print(klines[-1][0])

                return c_symbol

        except:
            # printing stack trace
            traceback.print_exc()
    else:
        print(c_symbol+': not connect')
        return 'ZZZ'


def _last_candle_receiver(args):
    api_key_last_candle = 'yours'
    api_secret_last_candle = 'yours'
    connected = False
    attempts = 2
    sleep(round(random.uniform(0.0, 0.5), 2))
    while attempts > 0:
        try:
            b_client = Client(api_key_last_candle, api_secret_last_candle)
            connected = True
            break
        except:
            # # printing stack trace
            # traceback.print_exc()
            attempts -= 1
            sleep(round(random.uniform(0.05, 0.5), 2))

    c_symbol, kl_1h_pass, binance_time_kl_buffer_5m, binance_time_kl_buffer_1h, incomplete_last_5m, incomplete_last_1h = args

    if connected:
        try:
            c_df_5m = pd.read_csv(c_symbol + '__5M.csv', index_col=0)
            interval_5m = b_client.KLINE_INTERVAL_5MINUTE
            if incomplete_last_5m:
                last_5m_cls_date = datetime.strptime(c_df_5m.index[-2], "%Y-%m-%d %H:%M:%S")
            else:
                last_5m_cls_date = datetime.strptime(c_df_5m.index[-1], "%Y-%m-%d %H:%M:%S")
            last_5m_cls_timestamp = int(1000*last_5m_cls_date.replace(tzinfo=timezone.utc).timestamp())
            # print(symbol+': '+str(last_5m_cls_date)+' '+str(last_5m_cls_timestamp))
            # start_5m = '15 min ago UTC'  # '1 day ago UTC' , "11 hours ago UTC" , '15 Dec, 2020' , '1 Jan, 2021'
            received = False
            while not received:
                try:
                    # Fetch klines for any date range and interval
                    klines_5m = b_client.get_historical_klines(c_symbol, interval_5m, last_5m_cls_timestamp)  # ,end)
                    received = True
                except:
                    sleep(round(random.uniform(0.05, 0.5), 2))

            if kl_1h_pass:
                c_df_1h = pd.read_csv(c_symbol + '__1H.csv', index_col=0)
                interval_1h = b_client.KLINE_INTERVAL_1HOUR
                if incomplete_last_1h:
                    last_1h_cls_date = datetime.strptime(c_df_1h.index[-2], "%Y-%m-%d %H:%M:%S")
                else:
                    last_1h_cls_date = datetime.strptime(c_df_1h.index[-1], "%Y-%m-%d %H:%M:%S")
                last_1h_cls_timestamp = int(1000*last_1h_cls_date.replace(tzinfo=timezone.utc).timestamp())
                # start_1h = '3 hours ago UTC'  # '1 day ago UTC' , "11 hours ago UTC" , '15 Dec, 2020' , '1 Jan, 2021'
                received = False
                while not received:
                    try:
                        # Fetch klines for any date range and interval
                        klines_1h = b_client.get_historical_klines(c_symbol, interval_1h, last_1h_cls_timestamp)  # ,end)
                        received = True
                    except:
                        sleep(round(random.uniform(0.1, 0.5), 2))

            # delete unwanted data - just keep date, open, high, low, close, volume
            for line in klines_5m:
                del line[11]
                # del line[10]
                del line[9]
                # del line[8]
                # del line[7]
                del line[6]
                # del line[1]

            # option 4 - create a Pandas DataFrame and export to CSV
            c_df2_5m = pd.DataFrame(klines_5m, columns=['openTime', 'open', 'high', 'low', 'close', 'volume',
                                                    'QuoteAssetVolume', 'NumOfTrades', 'TakerBuyQuoteAssetVolume'])
            # c_df2_5m = pd.DataFrame(klines_5m, columns=['openTime', 'open', 'high', 'low', 'close', 'volume'])
            c_df2_5m.set_index('openTime', inplace=True)
            c_df2_5m = c_df2_5m.loc[ :(binance_time_kl_buffer_5m-1)*5*60*1000 , : ]
            # print(c_df2_5m)
            c_df2_5m.index = pd.to_datetime(c_df2_5m.index, unit='ms')
            c_df2_5m['closeTime'] = pd.to_datetime(c_df2_5m.index.values) + timedelta(minutes=5)
            c_df2_5m.set_index('closeTime', inplace=True)
            c_df2_5m['open'] = pd.to_numeric( c_df2_5m.open, downcast = 'float' )
            c_df2_5m['high'] = pd.to_numeric(c_df2_5m.high, downcast='float')
            c_df2_5m['low'] = pd.to_numeric(c_df2_5m.low, downcast='float')
            c_df2_5m['close'] = pd.to_numeric(c_df2_5m.close, downcast='float')
            c_df2_5m['volume'] = pd.to_numeric(c_df2_5m.volume, downcast='float')

            c_df2_5m['NumOfTrades'] = pd.to_numeric(c_df2_5m.NumOfTrades, downcast='integer')
            c_df2_5m['QuoteAssetVolume'] = pd.to_numeric(c_df2_5m.QuoteAssetVolume, downcast='float')
            c_df2_5m['TakerBuyQuoteAssetVolume'] = pd.to_numeric(c_df2_5m.TakerBuyQuoteAssetVolume, downcast='float')
            c_df2_5m['TakerSellQuoteAssetVolume'] = c_df2_5m['QuoteAssetVolume'] - c_df2_5m['TakerBuyQuoteAssetVolume']
            # print(c_df2_5m)
            # c_df_5m = pd.read_csv(c_symbol + '__5M.csv', index_col=0)
            # c_df_5m = c_df_5m.loc[ :datetime.utcfromtimestamp( (binance_time_kl_buffer_5m-2)*5*60*1000 ) , : ]
            # print(c_df_5m)
            if incomplete_last_5m:
                # del the last and incomplete kl
                c_df_5m = c_df_5m.iloc[:-1]
            else:
                # del the first kl
                c_df_5m = c_df_5m.iloc[1:]
            # print(c_df_5m)
            c_df_5m = c_df_5m.append(c_df2_5m)
            # print(c_df_5m)
            # export DataFrame to csv
            c_df_5m.to_csv(c_symbol + '__5M.csv')

            last_5m_more_1 = True if (len(c_df2_5m) > 1) else False
            last_1h_more_1 = False

            if kl_1h_pass:
                for line in klines_1h:
                    del line[11]
                    # del line[10]
                    del line[9]
                    # del line[8]
                    # del line[7]
                    del line[6]
                    # del line[1]
                c_df2_1h = pd.DataFrame(klines_1h, columns=['openTime', 'open', 'high', 'low', 'close', 'volume',
                                                        'QuoteAssetVolume', 'NumOfTrades', 'TakerBuyQuoteAssetVolume'])
                # c_df2_1h = pd.DataFrame(klines_1h, columns=['openTime', 'open', 'high', 'low', 'close', 'volume'])
                c_df2_1h.set_index('openTime', inplace=True)
                c_df2_1h = c_df2_1h.loc[ :(binance_time_kl_buffer_1h-1)*60*60*1000 , : ]
                c_df2_1h.index = pd.to_datetime(c_df2_1h.index, unit='ms')
                c_df2_1h['closeTime'] = pd.to_datetime(c_df2_1h.index.values) + timedelta(hours=1)
                c_df2_1h.set_index('closeTime', inplace=True)
                c_df2_1h['open'] = pd.to_numeric(c_df2_1h.open, downcast='float')
                c_df2_1h['high'] = pd.to_numeric(c_df2_1h.high, downcast='float')
                c_df2_1h['low'] = pd.to_numeric(c_df2_1h.low, downcast='float')
                c_df2_1h['close'] = pd.to_numeric(c_df2_1h.close, downcast='float')
                c_df2_1h['volume'] = pd.to_numeric(c_df2_1h.volume, downcast='float')

                c_df2_1h['NumOfTrades'] = pd.to_numeric(c_df2_1h.NumOfTrades, downcast='integer')
                c_df2_1h['QuoteAssetVolume'] = pd.to_numeric(c_df2_1h.QuoteAssetVolume, downcast='float')
                c_df2_1h['TakerBuyQuoteAssetVolume'] = pd.to_numeric(c_df2_1h.TakerBuyQuoteAssetVolume, downcast='float')
                c_df2_1h['TakerSellQuoteAssetVolume'] = c_df2_1h['QuoteAssetVolume'] - c_df2_1h['TakerBuyQuoteAssetVolume']

                # c_df_1h = pd.read_csv(c_symbol + '__1H.csv', index_col=0)
                if incomplete_last_1h:
                    # del the last and incomplete kl
                    c_df_1h = c_df_1h.iloc[:-1]
                else:
                    # del the first kl
                    c_df_1h = c_df_1h.iloc[1:]

                last_1h_more_1 = True if (len(c_df2_1h) > 1) else False
                c_df_1h = c_df_1h.append(c_df2_1h)
                # export DataFrame to csv
                c_df_1h.to_csv(c_symbol + '__1H.csv')

            return c_symbol, [last_5m_more_1, last_1h_more_1]
        except:
            # printing stack trace
            traceback.print_exc()
    else:
        print(c_symbol+': not connect')
        return c_symbol, 'ZZZ'


def topup_bnb(bnb_client, min_balance=0.02, top_up=0.1):
    try:
        bnb_balance = bnb_client.get_asset_balance(asset='BNB')
        bnb_balance = float(bnb_balance['free'])
        if bnb_balance < min_balance:
            qty = round(top_up - bnb_balance, 3)
            print(qty)
            try:
                bnb_order = bnb_client.order_market_buy(symbol='BNBUSDT', quantity=qty)
                print('BNB order: ' + str(bnb_order))
                return bnb_order
            except:
                # printing stack trace
                traceback.print_exc()
                print('BNB balance is low and cannot be bought. bnb_balance: ' + str(bnb_balance))
                return False
        print('BNB balance is good. bnb_balance: ' + str(bnb_balance))
        return False
    except:
        # printing stack trace
        traceback.print_exc()
        print('BNB balance cannot be fetched')
        return False


if __name__ == '__main__':
    Api_Key = 'yours'
    Api_Secret = 'yours'
    client = Client(Api_Key, Api_Secret)
    server_time_dic = client.get_server_time()
    print(server_time_dic)
    status = client.get_system_status()
    print(status)

    # # get all symbol prices
    # x = []
    # symbol_prices = client.get_all_tickers()
    # for symbol_price in symbol_prices:
    #     if symbol_price['symbol'].find('USDT') > 2:
    #         x.append(symbol_price['symbol'])
    # print(x)
    # sleep(5)

    # SymbolsUSDT = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'BCCUSDT', 'NEOUSDT', 'LTCUSDT', 'QTUMUSDT', 'ADAUSDT', 'XRPUSDT', 'EOSUSDT', 'TUSDUSDT', 'IOTAUSDT', 'XLMUSDT', 'ONTUSDT', 'TRXUSDT', 'ETCUSDT', 'ICXUSDT', 'VENUSDT', 'NULSUSDT', 'VETUSDT', 'PAXUSDT', 'BCHABCUSDT', 'BCHSVUSDT', 'USDCUSDT', 'LINKUSDT', 'WAVESUSDT', 'BTTUSDT', 'USDSUSDT', 'ONGUSDT', 'HOTUSDT', 'ZILUSDT', 'ZRXUSDT', 'FETUSDT', 'BATUSDT', 'XMRUSDT', 'ZECUSDT', 'IOSTUSDT', 'CELRUSDT', 'DASHUSDT', 'NANOUSDT', 'OMGUSDT', 'THETAUSDT', 'ENJUSDT', 'MITHUSDT', 'MATICUSDT', 'ATOMUSDT', 'TFUELUSDT', 'ONEUSDT', 'FTMUSDT', 'ALGOUSDT', 'USDSBUSDT', 'GTOUSDT', 'ERDUSDT', 'DOGEUSDT', 'DUSKUSDT', 'ANKRUSDT', 'WINUSDT', 'COSUSDT', 'PUNDIXUSDT', 'COCOSUSDT', 'MTLUSDT', 'TOMOUSDT', 'PERLUSDT', 'DENTUSDT', 'MFTUSDT', 'KEYUSDT', 'STORMUSDT', 'DOCKUSDT', 'WANUSDT', 'FUNUSDT', 'CVCUSDT', 'CHZUSDT', 'BANDUSDT', 'BUSDUSDT', 'BEAMUSDT', 'XTZUSDT', 'RENUSDT', 'RVNUSDT', 'HBARUSDT', 'NKNUSDT', 'STXUSDT', 'KAVAUSDT', 'ARPAUSDT', 'IOTXUSDT', 'RLCUSDT', 'MCOUSDT', 'CTXCUSDT', 'BCHUSDT', 'TROYUSDT', 'VITEUSDT', 'FTTUSDT', 'EURUSDT', 'OGNUSDT', 'DREPUSDT', 'BULLUSDT', 'BEARUSDT', 'ETHBULLUSDT', 'ETHBEARUSDT', 'TCTUSDT', 'WRXUSDT', 'BTSUSDT', 'LSKUSDT', 'BNTUSDT', 'LTOUSDT', 'EOSBULLUSDT', 'EOSBEARUSDT', 'XRPBULLUSDT', 'XRPBEARUSDT', 'STRATUSDT', 'AIONUSDT', 'MBLUSDT', 'COTIUSDT', 'BNBBULLUSDT', 'BNBBEARUSDT', 'STPTUSDT', 'WTCUSDT', 'DATAUSDT', 'XZCUSDT', 'SOLUSDT', 'CTSIUSDT', 'HIVEUSDT', 'CHRUSDT', 'BTCUPUSDT', 'BTCDOWNUSDT', 'GXSUSDT', 'ARDRUSDT', 'LENDUSDT', 'MDTUSDT', 'STMXUSDT', 'KNCUSDT', 'REPUSDT', 'LRCUSDT', 'PNTUSDT', 'COMPUSDT', 'BKRWUSDT', 'ZENUSDT', 'SNXUSDT', 'ETHUPUSDT', 'ETHDOWNUSDT', 'ADAUPUSDT', 'ADADOWNUSDT', 'LINKUPUSDT', 'LINKDOWNUSDT', 'VTHOUSDT', 'DGBUSDT', 'GBPUSDT', 'SXPUSDT', 'MKRUSDT', 'DAIUSDT', 'DCRUSDT', 'STORJUSDT', 'BNBUPUSDT', 'BNBDOWNUSDT', 'XTZUPUSDT', 'XTZDOWNUSDT', 'MANAUSDT', 'AUDUSDT', 'YFIUSDT', 'BALUSDT', 'BLZUSDT', 'IRISUSDT', 'KMDUSDT', 'JSTUSDT', 'SRMUSDT', 'ANTUSDT', 'CRVUSDT', 'SANDUSDT', 'OCEANUSDT', 'NMRUSDT', 'DOTUSDT', 'LUNAUSDT', 'RSRUSDT', 'PAXGUSDT', 'WNXMUSDT', 'TRBUSDT', 'BZRXUSDT', 'SUSHIUSDT', 'YFIIUSDT', 'KSMUSDT', 'EGLDUSDT', 'DIAUSDT', 'RUNEUSDT', 'FIOUSDT', 'UMAUSDT', 'EOSUPUSDT', 'EOSDOWNUSDT', 'TRXUPUSDT', 'TRXDOWNUSDT', 'XRPUPUSDT', 'XRPDOWNUSDT', 'DOTUPUSDT', 'DOTDOWNUSDT', 'BELUSDT', 'WINGUSDT', 'LTCUPUSDT', 'LTCDOWNUSDT', 'UNIUSDT', 'NBSUSDT', 'OXTUSDT', 'SUNUSDT', 'AVAXUSDT', 'HNTUSDT', 'FLMUSDT', 'UNIUPUSDT', 'UNIDOWNUSDT', 'ORNUSDT', 'UTKUSDT', 'XVSUSDT', 'ALPHAUSDT', 'AAVEUSDT', 'NEARUSDT', 'SXPUPUSDT', 'SXPDOWNUSDT', 'FILUSDT', 'FILUPUSDT', 'FILDOWNUSDT', 'YFIUPUSDT', 'YFIDOWNUSDT', 'INJUSDT', 'AUDIOUSDT', 'CTKUSDT', 'BCHUPUSDT', 'BCHDOWNUSDT', 'AKROUSDT', 'AXSUSDT', 'HARDUSDT', 'DNTUSDT', 'STRAXUSDT', 'UNFIUSDT', 'ROSEUSDT', 'AVAUSDT', 'XEMUSDT', 'AAVEUPUSDT', 'AAVEDOWNUSDT', 'SKLUSDT', 'SUSDUSDT', 'SUSHIUPUSDT', 'SUSHIDOWNUSDT', 'XLMUPUSDT', 'XLMDOWNUSDT', 'GRTUSDT', 'JUVUSDT', 'PSGUSDT', '1INCHUSDT', 'REEFUSDT', 'ATMUSDT', 'ASRUSDT', 'CELOUSDT', 'RIFUSDT', 'BTCSTUSDT', 'TRUUSDT', 'CKBUSDT', 'TWTUSDT', 'FIROUSDT', 'LITUSDT', 'SFPUSDT', 'DODOUSDT', 'CAKEUSDT', 'ACMUSDT', 'BADGERUSDT', 'FISUSDT']
    # SymbolsUSDT.sort()
    # print(SymbolsUSDT)
    # sleep(5)

    # SymbolsUSDT = ['ALPHAUSDT']
    # SymbolsUSDT = ['ALPHAUSDT', 'MATICUSDT', 'XRPUSDT']
    # SymbolsUSDT = ['MTLUSDT', 'GRTUSDT', 'PUNDIXUSDT', 'STORJUSDT', 'AUDIOUSDT', 'FUNUSDT'] # 'BNBUSDT', 'BTCUSDT', 'ETHUSDT',
    # SymbolsUSDT = ['1INCHUSDT', 'AAVEUSDT', 'ADAUSDT', 'ALGOUSDT', 'ALPHAUSDT', 'ANKRUSDT', 'ARPAUSDT', 'ATOMUSDT', 'AUDIOUSDT', 'AVAUSDT', 'AVAXUSDT', 'AXSUSDT', 'BALUSDT', 'BANDUSDT', 'BATUSDT', 'BEAMUSDT', 'BELUSDT', 'BLZUSDT']
    # SymbolsUSDT = ['1INCHUSDT', 'AAVEUSDT', 'ADAUSDT', 'ALGOUSDT', 'ALPHAUSDT', 'ANKRUSDT', 'ARPAUSDT', 'ATOMUSDT', 'AUDIOUSDT', 'AVAUSDT', 'AVAXUSDT', 'AXSUSDT', 'BALUSDT', 'BANDUSDT', 'BATUSDT', 'BEAMUSDT', 'BELUSDT', 'BLZUSDT', 'BNTUSDT', 'BZRXUSDT', 'CELOUSDT', 'CELRUSDT', 'COMPUSDT', 'CRVUSDT', 'CTKUSDT', 'CTSIUSDT', 'CTXCUSDT', 'CVCUSDT', 'DASHUSDT', 'DATAUSDT', 'DCRUSDT', 'DNTUSDT', 'DOGEUSDT', 'DOTUSDT', 'DREPUSDT', 'DUSKUSDT', 'ENJUSDT', 'EOSUSDT', 'FTMUSDT', 'FUNUSDT', 'GRTUSDT', 'GTOUSDT', 'HARDUSDT', 'HBARUSDT', 'HIVEUSDT', 'ICXUSDT', 'INJUSDT', 'IOSTUSDT', 'IRISUSDT', 'KAVAUSDT', 'KNCUSDT', 'KSMUSDT', 'LINKUSDT', 'LRCUSDT', 'LTCUSDT', 'LUNAUSDT', 'MANAUSDT', 'MBLUSDT', 'MITHUSDT', 'MKRUSDT', 'NANOUSDT', 'NBSUSDT', 'NEARUSDT', 'NEOUSDT', 'NKNUSDT', 'NMRUSDT', 'PUNDIXUSDT', 'NULSUSDT', 'OGNUSDT', 'OMGUSDT', 'ONEUSDT', 'ONGUSDT', 'ONTUSDT', 'ORNUSDT', 'OXTUSDT', 'PERLUSDT', 'REEFUSDT', 'RENUSDT', 'REPUSDT', 'RLCUSDT', 'ROSEUSDT', 'RUNEUSDT', 'SANDUSDT', 'SKLUSDT', 'SNXUSDT', 'SOLUSDT', 'SRMUSDT', 'STPTUSDT', 'SUNUSDT', 'SUSHIUSDT', 'SXPUSDT', 'TFUELUSDT', 'THETAUSDT', 'TRBUSDT', 'TROYUSDT', 'UMAUSDT', 'UNFIUSDT', 'UNIUSDT', 'VETUSDT', 'VITEUSDT', 'VTHOUSDT', 'WAVESUSDT', 'WINGUSDT', 'WINUSDT', 'XRPUSDT', 'XTZUSDT', 'XVSUSDT', 'YFIIUSDT', 'YFIUSDT', 'ZECUSDT', 'ZENUSDT', 'ZILUSDT', 'ZRXUSDT']

    SymbolsUSDT = ['1INCHUSDT', 'AAVEUSDT', 'ACMUSDT', 'ADAUSDT', 'AIONUSDT', 'AKROUSDT', 'ALGOUSDT', 'ALPHAUSDT', 'ANKRUSDT', 'ANTUSDT', 'ARDRUSDT', 'ARPAUSDT', 'ASRUSDT', 'ATMUSDT', 'ATOMUSDT', 'AUDIOUSDT', 'AUDUSDT', 'AVAUSDT', 'AVAXUSDT', 'AXSUSDT', 'BADGERUSDT', 'BALUSDT', 'BANDUSDT', 'BATUSDT', 'BCHUSDT', 'BEAMUSDT', 'BELUSDT', 'BLZUSDT', 'BNBUSDT', 'BNTUSDT', 'BTCUSDT', 'BTSUSDT', 'BTTUSDT', 'BZRXUSDT', 'CAKEUSDT', 'CELOUSDT', 'CELRUSDT', 'CHRUSDT', 'CHZUSDT', 'CKBUSDT', 'COCOSUSDT', 'COMPUSDT', 'COSUSDT', 'COTIUSDT', 'CRVUSDT', 'CTKUSDT', 'CTSIUSDT', 'CTXCUSDT', 'CVCUSDT', 'DASHUSDT', 'DATAUSDT', 'DCRUSDT', 'DENTUSDT', 'DGBUSDT', 'DIAUSDT', 'DNTUSDT', 'DOCKUSDT', 'DODOUSDT', 'DOGEUSDT', 'DOTUSDT', 'DREPUSDT', 'DUSKUSDT', 'EGLDUSDT', 'ENJUSDT', 'EOSUSDT', 'ETCUSDT', 'ETHUSDT', 'FETUSDT', 'FILUSDT', 'FIOUSDT', 'FIROUSDT', 'FISUSDT', 'FLMUSDT', 'FTMUSDT', 'FTTUSDT', 'FUNUSDT', 'GRTUSDT', 'GTOUSDT', 'GXSUSDT', 'HARDUSDT', 'HBARUSDT', 'HIVEUSDT', 'HNTUSDT', 'HOTUSDT', 'ICXUSDT', 'INJUSDT', 'IOSTUSDT', 'IOTAUSDT', 'IOTXUSDT', 'IRISUSDT', 'JSTUSDT', 'JUVUSDT', 'KAVAUSDT', 'KEYUSDT', 'KMDUSDT', 'KNCUSDT', 'KSMUSDT', 'LINKUSDT', 'LITUSDT', 'LRCUSDT', 'LSKUSDT', 'LTCUSDT', 'LTOUSDT', 'LUNAUSDT', 'MANAUSDT', 'MATICUSDT', 'MBLUSDT', 'MDTUSDT', 'MFTUSDT', 'MITHUSDT', 'MKRUSDT', 'MTLUSDT', 'NANOUSDT', 'NBSUSDT', 'NEARUSDT', 'NEOUSDT', 'NKNUSDT', 'NMRUSDT', 'PUNDIXUSDT', 'NULSUSDT', 'OCEANUSDT', 'OGNUSDT', 'OMGUSDT', 'ONEUSDT', 'ONGUSDT', 'ONTUSDT', 'ORNUSDT', 'OXTUSDT', 'PAXGUSDT', 'PAXUSDT', 'PERLUSDT', 'PNTUSDT', 'PSGUSDT', 'QTUMUSDT', 'REEFUSDT', 'RENUSDT', 'REPUSDT', 'RIFUSDT', 'RLCUSDT', 'ROSEUSDT', 'RSRUSDT', 'RUNEUSDT', 'RVNUSDT', 'SANDUSDT', 'SFPUSDT', 'SKLUSDT', 'SNXUSDT', 'SOLUSDT', 'SRMUSDT', 'STMXUSDT', 'STORJUSDT', 'STPTUSDT', 'STRAXUSDT', 'STXUSDT', 'SUNUSDT', 'SUSDUSDT', 'SUSHIUSDT', 'SXPUSDT', 'TCTUSDT', 'TFUELUSDT', 'THETAUSDT', 'TOMOUSDT', 'TRBUSDT', 'TROYUSDT', 'TRUUSDT', 'TRXUSDT', 'TWTUSDT', 'UMAUSDT', 'UNFIUSDT', 'UNIUSDT', 'UTKUSDT', 'VETUSDT', 'VITEUSDT', 'VTHOUSDT', 'WANUSDT', 'WAVESUSDT', 'WINGUSDT', 'WINUSDT', 'WNXMUSDT', 'WRXUSDT', 'WTCUSDT', 'XEMUSDT', 'XLMUSDT', 'XMRUSDT', 'XRPUSDT', 'XTZUSDT', 'XVSUSDT', 'YFIIUSDT', 'YFIUSDT', 'ZECUSDT', 'ZENUSDT', 'ZILUSDT', 'ZRXUSDT']

    # SymbolsUSDT2 = ['1INCHUSDT', 'AAVEUSDT', 'ACMUSDT', 'ADAUSDT', 'AIONUSDT', 'AKROUSDT', 'ALGOUSDT', 'ALPHAUSDT', 'ANKRUSDT', 'ANTUSDT', 'ARDRUSDT', 'ARPAUSDT', 'ASRUSDT', 'ATMUSDT', 'ATOMUSDT', 'AUDIOUSDT', 'AUDUSDT', 'AVAUSDT', 'AVAXUSDT', 'AXSUSDT', 'BADGERUSDT', 'BALUSDT', 'BANDUSDT', 'BATUSDT', 'BCCUSDT', 'BCHABCUSDT', 'BCHDOWNUSDT', 'BCHSVUSDT', 'BCHUPUSDT', 'BCHUSDT', 'BEAMUSDT', 'BEARUSDT', 'BELUSDT', 'BKRWUSDT', 'BLZUSDT', 'BNBBEARUSDT', 'BNBBULLUSDT', 'BNBDOWNUSDT', 'BNBUPUSDT', 'BNBUSDT', 'BNTUSDT', 'BTCDOWNUSDT', 'BTCSTUSDT', 'BTCUPUSDT', 'BTCUSDT', 'BTSUSDT', 'BTTUSDT', 'BULLUSDT', 'BUSDUSDT', 'BZRXUSDT', 'CAKEUSDT', 'CELOUSDT', 'CELRUSDT', 'CHRUSDT', 'CHZUSDT', 'CKBUSDT', 'COCOSUSDT', 'COMPUSDT', 'COSUSDT', 'COTIUSDT', 'CRVUSDT', 'CTKUSDT', 'CTSIUSDT', 'CTXCUSDT', 'CVCUSDT', 'DAIUSDT', 'DASHUSDT', 'DATAUSDT', 'DCRUSDT', 'DENTUSDT', 'DGBUSDT', 'DIAUSDT', 'DNTUSDT', 'DOCKUSDT', 'DODOUSDT', 'DOGEUSDT', 'DOTDOWNUSDT', 'DOTUPUSDT', 'DOTUSDT', 'DREPUSDT', 'DUSKUSDT', 'EGLDUSDT', 'ENJUSDT', 'EOSBEARUSDT', 'EOSBULLUSDT', 'EOSDOWNUSDT', 'EOSUPUSDT', 'EOSUSDT', 'ERDUSDT', 'ETCUSDT', 'ETHBEARUSDT', 'ETHBULLUSDT', 'ETHDOWNUSDT', 'ETHUPUSDT', 'ETHUSDT', 'EURUSDT', 'FETUSDT', 'FILDOWNUSDT', 'FILUPUSDT', 'FILUSDT', 'FIOUSDT', 'FIROUSDT', 'FISUSDT', 'FLMUSDT', 'FTMUSDT', 'FTTUSDT', 'FUNUSDT', 'GBPUSDT', 'GRTUSDT', 'GTOUSDT', 'GXSUSDT', 'HARDUSDT', 'HBARUSDT', 'HIVEUSDT', 'HNTUSDT', 'HOTUSDT', 'ICXUSDT', 'INJUSDT', 'IOSTUSDT', 'IOTAUSDT', 'IOTXUSDT', 'IRISUSDT', 'JSTUSDT', 'JUVUSDT', 'KAVAUSDT', 'KEYUSDT', 'KMDUSDT', 'KNCUSDT', 'KSMUSDT', 'LENDUSDT', 'LINKDOWNUSDT', 'LINKUPUSDT', 'LINKUSDT', 'LITUSDT', 'LRCUSDT', 'LSKUSDT', 'LTCDOWNUSDT', 'LTCUPUSDT', 'LTCUSDT', 'LTOUSDT', 'LUNAUSDT', 'MANAUSDT', 'MATICUSDT', 'MBLUSDT', 'MCOUSDT', 'MDTUSDT', 'MFTUSDT', 'MITHUSDT', 'MKRUSDT', 'MTLUSDT', 'NANOUSDT', 'NBSUSDT', 'NEARUSDT', 'NEOUSDT', 'NKNUSDT', 'NMRUSDT', 'PUNDIXUSDT', 'NULSUSDT', 'OCEANUSDT', 'OGNUSDT', 'OMGUSDT', 'ONEUSDT', 'ONGUSDT', 'ONTUSDT', 'ORNUSDT', 'OXTUSDT', 'PAXGUSDT', 'PAXUSDT', 'PERLUSDT', 'PNTUSDT', 'PSGUSDT', 'QTUMUSDT', 'REEFUSDT', 'RENUSDT', 'REPUSDT', 'RIFUSDT', 'RLCUSDT', 'ROSEUSDT', 'RSRUSDT', 'RUNEUSDT', 'RVNUSDT', 'SANDUSDT', 'SFPUSDT', 'SKLUSDT', 'SNXUSDT', 'SOLUSDT', 'SRMUSDT', 'STMXUSDT', 'STORJUSDT', 'STORMUSDT', 'STPTUSDT', 'STRATUSDT', 'STRAXUSDT', 'STXUSDT', 'SUNUSDT', 'SUSDUSDT', 'SUSHIDOWNUSDT', 'SUSHIUPUSDT', 'SUSHIUSDT', 'SXPDOWNUSDT', 'SXPUPUSDT', 'SXPUSDT', 'TCTUSDT', 'TFUELUSDT', 'THETAUSDT', 'TOMOUSDT', 'TRBUSDT', 'TROYUSDT', 'TRUUSDT', 'TRXDOWNUSDT', 'TRXUPUSDT', 'TRXUSDT', 'TUSDUSDT', 'TWTUSDT', 'UMAUSDT', 'UNFIUSDT', 'UNIDOWNUSDT', 'UNIUPUSDT', 'UNIUSDT', 'USDCUSDT', 'USDSBUSDT', 'USDSUSDT', 'UTKUSDT', 'VENUSDT', 'VETUSDT', 'VITEUSDT', 'VTHOUSDT', 'WANUSDT', 'WAVESUSDT', 'WINGUSDT', 'WINUSDT', 'WNXMUSDT', 'WRXUSDT', 'WTCUSDT', 'XEMUSDT', 'XLMDOWNUSDT', 'XLMUPUSDT', 'XLMUSDT', 'XMRUSDT', 'XRPBEARUSDT', 'XRPBULLUSDT', 'XRPDOWNUSDT', 'XRPUPUSDT', 'XRPUSDT', 'XTZDOWNUSDT', 'XTZUPUSDT', 'XTZUSDT', 'XVSUSDT', 'XZCUSDT', 'YFIDOWNUSDT', 'YFIIUSDT', 'YFIUPUSDT', 'YFIUSDT', 'ZECUSDT', 'ZENUSDT', 'ZILUSDT', 'ZRXUSDT']
    # diff = list(set(SymbolsUSDT2) - set(SymbolsUSDT))
    # diff.sort()
    # print(diff)
    # sleep(5)

    # info = client.get_symbol_info( SymbolsUSDT[0] )
    # print( info )

    usdt_balance_1st = client.get_asset_balance(asset='USDT')
    usdt_balance_1st = float(usdt_balance_1st['free'])
    max_num_active_investment = 3
    budget = usdt_balance_1st // (max_num_active_investment+0.5)
    print('usdt_balance: ' + str(usdt_balance_1st))

    print(len(SymbolsUSDT))

    # SymbolReadWriteLocks = {}
    # for symbol in SymbolsUSDT:
    #     SymbolReadWriteLocks[symbol] = threading.Lock()
    # sellLock = threading.Lock()
    # profitLock = threading.Lock()
    # binanceConnectionLock = threading.Lock()
    debugPrintLock = threading.Lock()

    Incomplete_last_5m = {}
    Incomplete_last_1h = {}

    available_symbol_1w_history = []
    available_symbol_open_time_w = {}
    num_threads_for_history = len(SymbolsUSDT)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads_for_history) as executor:
        results = executor.map(_1w_history_receiver, SymbolsUSDT)
    # with concurrent.futures.ProcessPoolExecutor(max_workers=9) as executor:
    #     results = executor.map(_1w_history_receiver, SymbolsUSDT)
    for symbol, open_time_w in results:
        if open_time_w != 'ZZZ':
            available_symbol_1w_history.append(symbol)
            available_symbol_open_time_w[symbol] = open_time_w
    print("1w History has just been Received :)")
    print('# of available_symbol_1w_history : ' + str(len(available_symbol_1w_history)))
    print('not available : ')
    notAvailable = list(set(SymbolsUSDT) - set(available_symbol_1w_history))
    print(notAvailable)

    # sleep(4)

    available_symbol_1d_history = []
    num_threads_for_history = len(available_symbol_1w_history)
    args = ((symbol, available_symbol_open_time_w[symbol]) for symbol in available_symbol_1w_history)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads_for_history) as executor:
    #     results = executor.map(_1d_history_receiver, args)
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        results = executor.map(_1d_history_receiver, args)
    for result in results:
        if result != 'ZZZ':
            available_symbol_1d_history.append(result)
    print("1d History has just been Received :)")
    print('# of available_symbol_1d_history : ' + str(len(available_symbol_1d_history)))
    print('not available : ')
    notAvailable = list(set(available_symbol_1w_history) - set(available_symbol_1d_history))
    print(notAvailable)

    # sleep(4)

    available_symbol_1h_history = []
    num_threads_for_history = len(available_symbol_1d_history)
    args = ((symbol, available_symbol_open_time_w[symbol]) for symbol in available_symbol_1d_history)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads_for_history) as executor:
    #     results = executor.map(_1h_history_receiver, args)
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        results = executor.map(_1h_history_receiver, args)
    for result in results:
        if result != 'ZZZ':
            available_symbol_1h_history.append(result)
    print("1h History has just been Received :)")
    print('# of available_symbol_1h_history : ' + str(len(available_symbol_1h_history)))
    print('not available : ')
    notAvailable = list(set(available_symbol_1d_history) - set(available_symbol_1h_history))
    print(notAvailable)

    # sleep(10)

    available_symbol_5m_history = []
    num_threads_for_history = len(available_symbol_1h_history)
    args = ((symbol, available_symbol_open_time_w[symbol]) for symbol in available_symbol_1h_history)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads_for_history) as executor:
    #     results = executor.map(_5m_history_receiver, args)
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        results = executor.map(_5m_history_receiver, args)
    for result in results:
        if result != 'ZZZ':
            available_symbol_5m_history.append(result)
            Incomplete_last_5m[result] = True
            Incomplete_last_1h[result] = True
    print("5m History has just been Received :)")
    print('# of available_symbol_5m_history : ' + str(len(available_symbol_5m_history)))
    print('not available : ')
    notAvailable = list(set(available_symbol_1h_history) - set(available_symbol_5m_history))
    print(notAvailable)    

    kl_60_minute = 60 * 1
    kl_5_minute = 5
    client = Client(Api_Key, Api_Secret)
    server_time_dic = client.get_server_time()
    binance_time_UTCms = server_time_dic['serverTime']
    Binance_Time_Kl_Buffer_1h = floor(binance_time_UTCms / (1000 * 60 * kl_60_minute))
    Binance_Time_Kl_Buffer_5m = floor(binance_time_UTCms / (1000 * 60 * kl_5_minute))

    d = {'symbol_transaction': ['X'], 'budget': [0.0], 'bought_price': [0.0], 'profit': [0], 'num_active_investment': [0],
         'active_investments': [np.nan]}
    profit_df = pd.DataFrame(data=d)
    # profit_df['active_investments'] = profit_df['active_investments'].astype(object)
    profit_df.to_csv('0profit0.csv')

    prev_potential_short = []
    prev_potential_long = []
    # kl_1h_start = False

    Kl_1h_Pass = False
    Kl_5m_pass = False
    while not Kl_5m_pass:
        received = False
        while not received:
            try:
                server_time_dic = client.get_server_time()
                received = True
            except:
                sleep(1)
        binance_time_UTCms = server_time_dic['serverTime']
        binance_time_kl_1h = floor(binance_time_UTCms / (1000 * 60 * kl_60_minute))
        binance_time_kl_5m = floor(binance_time_UTCms / (1000 * 60 * kl_5_minute))
        if binance_time_kl_5m >= (Binance_Time_Kl_Buffer_5m + 1):
            Kl_5m_pass = True
            Binance_Time_Kl_Buffer_5m = binance_time_kl_5m
            if binance_time_kl_1h >= (Binance_Time_Kl_Buffer_1h + 1):
                Kl_1h_Pass = True
                Binance_Time_Kl_Buffer_1h = binance_time_kl_1h
        else:
            sleep(2)

    print('\n\nGet Last 5m Candle')
    print(Kl_1h_Pass)

    available_last_kl = []
    num_threads_for_last_candle = len(available_symbol_5m_history)
    args = ((symbol, Kl_1h_Pass, Binance_Time_Kl_Buffer_5m, Binance_Time_Kl_Buffer_1h, Incomplete_last_5m[symbol], Incomplete_last_1h[symbol]) for symbol in available_symbol_5m_history)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads_for_last_candle) as executor:
        results = executor.map(_last_candle_receiver, args)
    # with concurrent.futures.ProcessPoolExecutor(max_workers=9) as executor:
    #     results = executor.map(_last_candle_receiver, args)
    for symbol, lasts in results:
        if lasts != 'ZZZ':
            available_last_kl.append(symbol)
            Incomplete_last_5m[symbol] = False
            if Kl_1h_Pass:
                Incomplete_last_1h[symbol] = False
    print("5m Last Candle has just been Received :)")
    print('# of available_last_kl : ' + str(len(available_last_kl)))
    print('not available : ')
    notAvailable = list(set(available_symbol_5m_history) - set(available_last_kl))
    print(notAvailable)

    last_kl_1h_features = {}
    trade_history = {}
    available_symbol_backtrader = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=9) as executor:
        results = executor.map(process_backtrader, available_last_kl)
    for symbol in results:
        if symbol != 'ZZZ':
            available_symbol_backtrader.append(symbol)
            last_kl_1h_features[symbol] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            trade_history[symbol] = {'bought1_sold0': 0, 'bought_price': 0.0, 'profit': 0.0, 'buy_time': 0}
    print('Backtrader has just been Done :)')
    print('# of available_symbol_backtrader : ' + str(len(available_symbol_backtrader)))
    print('not available : ')
    notAvailable = list(set(available_last_kl) - set(available_symbol_backtrader))
    print(notAvailable)

    # got_features = []
    # with concurrent.futures.ProcessPoolExecutor(max_workers=9) as executor:
    #     results = executor.map(process_get_features, available_symbol_backtrader)
    # for symbol in results:
    #     got_features.append(symbol)
    # print('getting features has just been Done :)')
    # print('# of got_features : ' + str(len(available_symbol_backtrader)))
    # print('not available : ')
    # notAvailable = list(set(available_symbol_backtrader) - set(got_features))
    # print(notAvailable)

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

    # Define the model
    filters = 84
    kernel_size = 6
    model_architecture = '3Conv1Df' + str(filters) + 'k' + str(kernel_size) + 'psame_MxPl1Do2_Dense128'
    DNN_activation = 'relu'
    last_activation = 'softmax'

    batch_size = 4 * num_label_type

    experiment_name = label_type + '_' + str(batch_size) + '_' + str(num_ds_row) + '_' + str(
        num_ds_col) + '_' + future_type + '_' + features_type + '_' + model_architecture + '_' + DNN_activation + '_' + last_activation

    model_output_dir = 'C:/Users/Mahdi/Documents/spyder/Crypto_Trader/DataDir/classification_output'
    checkpoint_filepath = os.path.join(model_output_dir, experiment_name, '04-0.8470439-0.8692896-0.3962670-0.3153605.h5')
    model = load_model(checkpoint_filepath)

    bought_assets = []

    while True:
        Kl_1h_Pass = False
        Kl_5m_pass = False
        # Kl_1m_pass = False
        while not Kl_5m_pass:
            received = False
            while not received:
                try:
                    server_time_dic = client.get_server_time()
                    received = True
                except:
                    sleep(1)
            binance_time_UTCms = server_time_dic['serverTime']
            binance_time_kl_1h = floor(binance_time_UTCms / (1000 * 60 * kl_60_minute))
            binance_time_kl_5m = floor(binance_time_UTCms / (1000 * 60 * kl_5_minute))
            if binance_time_kl_5m >= (Binance_Time_Kl_Buffer_5m + 1):
                Kl_5m_pass = True
                Binance_Time_Kl_Buffer_5m = binance_time_kl_5m
                if binance_time_kl_1h >= (Binance_Time_Kl_Buffer_1h + 1):
                    Kl_1h_Pass = True
                    Binance_Time_Kl_Buffer_1h = binance_time_kl_1h
            else:
                sleep(2)

        print('\n\nGet Last 5m Candle')
        timestamp = datetime.utcfromtimestamp(Binance_Time_Kl_Buffer_5m * kl_5_minute * 60)
        print(timestamp)
        # print('5m pass: ' + str(Kl_5m_pass))
        print('1h pass: '+str(Kl_1h_Pass) )

        available_1_last_kl = []
        available_more_1_last_kl = []
        num_threads_for_last_candle = len(available_symbol_backtrader)
        args = ((symbol, Kl_1h_Pass, Binance_Time_Kl_Buffer_5m, Binance_Time_Kl_Buffer_1h, Incomplete_last_5m[symbol], Incomplete_last_1h[symbol]) for symbol in available_symbol_backtrader)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads_for_last_candle) as executor:
            results = executor.map(_last_candle_receiver, args)
        # with concurrent.futures.ProcessPoolExecutor(max_workers=9) as executor:
        #     results = executor.map(_last_candle_receiver, args)
        for symbol, lasts in results:
            if lasts != 'ZZZ':
                if Kl_1h_Pass:
                    Incomplete_last_1h[symbol] = False
                Last_5m_more_1, Last_1h_more_1 = lasts
                if Last_5m_more_1 or Last_1h_more_1:
                    available_more_1_last_kl.append(symbol)
                else:
                    available_1_last_kl.append(symbol)
        print("5m Last Candle has just been Received :)")
        print('# of available_1_last_kl : ' + str(len(available_1_last_kl)))
        print('# of available_more_1_last_kl : ' + str(len(available_more_1_last_kl)))
        print('not available : ')
        notAvailable = list(set(available_symbol_backtrader) - set(available_1_last_kl) - set(available_more_1_last_kl))
        print(notAvailable)


        available_last_kl_features = []
        all_symbols_arr_feed = []
        args = ((symbol, Kl_1h_Pass, last_kl_1h_features[symbol], num_ds_row) for symbol in available_1_last_kl)
        with concurrent.futures.ProcessPoolExecutor(max_workers=11) as executor:
            results = executor.map(process_last_kl_get_featutes, args)
        for symbol, [volMA_1d_7kl, volMA_1h_8kl, volMA_1h_24kl, keltner_1h_lower25, keltner_1h_mid25, keltner_1h_upper25, keltner_1h_lower99, keltner_1h_mid99, keltner_1h_upper99], arr_feed in results:
            if symbol != 'ZZZ':
                available_last_kl_features.append(symbol)
                all_symbols_arr_feed.append(arr_feed)
                last_kl_1h_features[symbol] = [volMA_1d_7kl, volMA_1h_8kl, volMA_1h_24kl, keltner_1h_lower25, keltner_1h_mid25, keltner_1h_upper25, keltner_1h_lower99, keltner_1h_mid99, keltner_1h_upper99]
        print('last kl features has just been Done :)')
        print('# of available_last_kl_features : ' + str(len(available_last_kl_features)))
        print('not available : ')
        notAvailable = list(set(available_last_kl) - set(available_last_kl_features))
        print(notAvailable)

        model_feed = np.array(all_symbols_arr_feed, copy=False)
        model_output = model.predict(np.asarray(model_feed).astype('float32').reshape( len(available_last_kl_features), num_ds_row, num_ds_col))
        model_output_idx = np.argmax(model_output, axis=1)

        profit_df = pd.read_csv('0profit0.csv', index_col=0)
        # profit_df['active_investments'] = profit_df['active_investments'].astype(object)
        num_active_investment = profit_df['num_active_investment'].iat[0]

        symbol_modelOut = []

        print('\ncheck status:')
        for num, symbol in enumerate(available_last_kl_features):
            if trade_history[symbol]['bought1_sold0'] == 1 :
                print(symbol + ' prediction:' + trade_actions[model_output_idx[num]]+ ' modelOut:' + str(model_output[num,:]))
                # if model_output_idx[num] != 0 and model_output_idx[num] != 1:
                if (model_output_idx[num] == 1 and (model_output[num, 1] >= 0.55 or model_output[num, 0] <= 0.34)) or ( model_output_idx[num] == 2 and (model_output[num, 2] >= 0.55 or model_output[num, 0] <= 0.34)):
                    received = False
                    while not received:
                        try:
                            last_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                            received = True
                        except:
                            sleep(0.5)
                    # also consider the commission of buy and sell
                    profit = 0.9985*(1.0 + (last_price - trade_history[symbol]['bought_price']) / trade_history[symbol]['bought_price']) - 1.0
                    trade_history[symbol]['bought1_sold0'] = 0

                    num_active_investment -= 1
                    bought_assets.remove(symbol)
                    # profit_df.iat[0, profit_df.columns.get_loc('active_investments')] = str(profit_df.iloc[0]['active_investments']).replace(symbol + ',', '')
                    # profit_df.iat[0, profit_df.columns.get_loc('active_investments')] = str(profit_df.iloc[0]['active_investments']).replace(',' + symbol, '')
                    # profit_df.iat[0, profit_df.columns.get_loc('active_investments')] = str(profit_df.iloc[0]['active_investments']).replace(symbol, '')

                    cd = {'symbol_transaction': [symbol+'_BT_'+str(trade_history[symbol]['buy_time'])+'_ST_'+str(timestamp)],
                          'budget': [budget], 'bought_price': [trade_history[symbol]['bought_price']], 'profit': [profit]}
                    c_profit_df = pd.DataFrame(data=cd)
                    profit_df = profit_df.append(c_profit_df)

            if trade_history[symbol]['bought1_sold0'] == 0 and model_output_idx[num] == 0 :
                symbol_modelOut.append([symbol, model_output[num, : ]])
                # print(symbol + ' score: ' + str(model_output[num, 0]))

        print('\nShort Long predictions:')
        symbol_modelOut.sort(key=lambda x: x[1][0], reverse=True)
        for symbol, modelOut in symbol_modelOut:
            print(symbol+' modelOut: '+str(modelOut))

        print('\nbuy these:')
        for symbol, modelOut in symbol_modelOut:
            if (num_active_investment < max_num_active_investment) and (modelOut[0] >= 0.75):
                received = False
                while not received:
                    try:
                        last_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                        received = True
                    except:
                        sleep(0.5)
                print('buying '+symbol)
                trade_history[symbol]['bought_price'] = last_price
                trade_history[symbol]['bought1_sold0'] = 1
                trade_history[symbol]['buy_time'] = timestamp

                bought_assets.append(symbol)

                num_active_investment += 1
                # if not profit_df['active_investments'].notna().values.any():
                #     profit_df.iat[0, profit_df.columns.get_loc('active_investments')] = symbol
                # else:
                #     profit_df.iat[0, profit_df.columns.get_loc('active_investments')] = str(profit_df.iloc[0]['active_investments']) + ',' + symbol

        # if len(bought_assets) == 0:
        #     usdt_balance = client.get_asset_balance(asset='USDT')
        #     usdt_balance = float(usdt_balance['free'])
        #     budget = usdt_balance / (max_num_active_investment + 0.5)
        #     print('usdt_balance: ' + str(usdt_balance))
        #     total_profit = (usdt_balance - usdt_balance_1st) / usdt_balance_1st

        profit_df.at[profit_df.index[0], 'num_active_investment'] = num_active_investment
        # profit_df.iat[0, profit_df.columns.get_loc('active_investments')] = bought_assets
        profit_df.at[profit_df.index[0], 'active_investments'] = profit_df['active_investments'].astype(object)
        # profit_df.at['X', 'active_investments'] = str(bought_assets)
        profit_df.to_csv('0profit0.csv')
        print('Total profit = ' + str(profit_df['profit'].sum()))
        print('# of active investments: ' + str(num_active_investment))
        print('active investments: ' + str(bought_assets))

        if len(available_more_1_last_kl) >= 1:
            available_more_1_last_kl_features = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=9) as executor:
                results = executor.map(process_backtrader, available_more_1_last_kl)
            for symbol in results:
                if symbol != 'ZZZ':
                    available_more_1_last_kl_features.append(symbol)
                    last_kl_1h_features[symbol] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            print('last "more than 1" kl features has just been Done :)')
            print('# of available_more_1_last_kl_features : ' + str(len(available_more_1_last_kl_features)))
            print('not available : ')
            notAvailable = list(set(available_more_1_last_kl) - set(available_more_1_last_kl_features))
            print(notAvailable)
        
        # # if Kl_1h_Pass:
        # #     kl_1h_start = True
        # # else:
        # #     kl_1h_start = False
        #
        # for symbol in available_last_kl:
        #     with SymbolReadWriteLocks[symbol]:
        #         currency_df = pd.read_csv(symbol + '__H.csv', index_col=0)
        #     # Here is how we can calculate the SuperTrend & ATR indicators
        #     atr10d = btalib.atr(currency_df, _period=1, period=100)
        #     currency_df['ATR'] = atr10d.df.atr
        #     with SymbolReadWriteLocks[symbol]:
        #         # export DataFrame to csv
        #         currency_df.to_csv(symbol + '__H.csv')
        #
        # args = ((symbol, first_time_h[symbol]) for symbol in available_last_kl)
        # num_threads_for_last_candle = len(available_last_kl)
        # if num_threads_for_last_candle > 0:
        #     with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads_for_last_candle) as executor:
        #         symbols = executor.map(threader_h_last_candle_atr_sprtrend, args)
        #
        #     for symbol in symbols:
        #         first_time_h[symbol] = False
        #         with SymbolReadWriteLocks[symbol]:
        #             currency_df = pd.read_csv(symbol + '__H.csv', index_col=0)
        #
        #         if currency_df.iloc[-1]['transaction_state'] == 0:
        #             currency_df.iloc[
        #                 -1, currency_df.columns.get_loc('transaction_state')] = 0  # have not been bought
        #
        #             if (currency_df.iloc[-1]['SelfNrATR'] > -1.0 or currency_df.iloc[-1]['score_short'] > 1000.0 or
        #                 currency_df.iloc[-1]['score_long'] > 1000.0) and (
        #                     currency_df.iloc[-1]['BySlDfNowPerc'] > 0.0 and currency_df.iloc[-1][
        #                 'BySlDfSlp2kl'] >= 4.0):
        #                 if currency_df.iloc[-1]['Price_st'] == 3 or currency_df.iloc[-1]['Price_st'] == 4 or \
        #                         currency_df.iloc[-1]['Price_st'] == 5:
        #                     if (currency_df.iloc[-1]['ByPr_st'] >= 0 and currency_df.iloc[-1][
        #                         'BySlDf_st'] >= 0) or (currency_df.iloc[-1]['ByPr_st'] <= 0 and (
        #                             currency_df.iloc[-1]['BySlDf_st'] >= 2 or currency_df.iloc[-2][
        #                         'BySlDf_st'] >= 2)):
        #                         HourBuyCandidates_sy.append(symbol)
        #                         HourBuyCandidates_sy_sc_pos.append(
        #                             [symbol, currency_df.iloc[-1]['score_short'], 'short'])
        #
        #             elif (Kl_1h_Pass and currency_df.iloc[-1]['score_short'] > -100.0 and
        #                   currency_df.iloc[-1]['score_long'] > 0.0) \
        #                     and (
        #                     currency_df.iloc[-1]['BySlDfNowPerc'] > 0.0 and currency_df.iloc[-1]['ClsSlp2kl'] > 0.0) \
        #                     and (currency_df.iloc[-1]['BySlDf_st'] >= 2 or currency_df.iloc[-2]['BySlDf_st'] >= 2) \
        #                     and (currency_df.iloc[-1]['ClsSlpSlp16kl4kl'] > 0.0 and currency_df.iloc[-1][
        #                 'ClsSlpSlp32kl4kl'] > 0.0) \
        #                     and ((currency_df.iloc[-4]['ClsSlpSlp16kl4kl'] < 0.0 or currency_df.iloc[-3][
        #                 'ClsSlpSlp16kl4kl'] < 0.0 or currency_df.iloc[-2]['ClsSlpSlp16kl4kl'] < 0.0)
        #                          or (currency_df.iloc[-4]['ClsSlpSlp32kl4kl'] < 0.0 or currency_df.iloc[-3][
        #                         'ClsSlpSlp32kl4kl'] < 0.0 or currency_df.iloc[-2]['ClsSlpSlp32kl4kl'] < 0.0)):
        #                 HourBuyCandidates_sy.append(symbol)
        #                 HourBuyCandidates_sy_sc_pos.append([symbol, currency_df.iloc[-1]['score_long'], 'long'])
        #
        #         with SymbolReadWriteLocks[symbol]:
        #             # export DataFrame to csv
        #             currency_df.to_csv(symbol + '__H.csv')
        # else:
        #     print('poor connection, hour last candle cannot be received !!')
        #
        # print('HourBuyCandidates are : ' + str(HourBuyCandidates_sy_sc_pos))
        # # num_threads_for_d_last_candle = len(HourBuyCandidates_sy)
        # num_threads_for_d_last_candle = len(available_symbol_1h_history)
        # if num_threads_for_d_last_candle > 0:
        #     AvailableDayLastCandle = []
        #     with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads_for_d_last_candle) as executor:
        #         # results = executor.map( threader_d_last_candle_receiver, HourBuyCandidates_sy )
        #         results = executor.map(threader_d_last_candle_receiver, available_symbol_1h_history)
        #     for result in results:
        #         if result != 'ZZZ':
        #             AvailableDayLastCandle.append(result)
        #     notAvailable = list(set(available_symbol_1h_history) - set(AvailableDayLastCandle))
        #     print("Day Last Candle has just been Received :)")
        #     print('# of AvailableDayLastCandle : ' + str(len(AvailableDayLastCandle)))
        #     print('not available : ')
        #     print(notAvailable)
        #
        #     for symbol in AvailableDayLastCandle:
        #         with SymbolReadWriteLocks[symbol]:
        #             currency_df = pd.read_csv(symbol + '__D.csv', index_col=0)
        #         # Here is how we can calculate the SuperTrend & ATR indicators
        #         atr10d = btalib.atr(currency_df, _period=1, period=60)
        #         currency_df['ATR'] = atr10d.df.atr
        #         with SymbolReadWriteLocks[symbol]:
        #             # export DataFrame to csv
        #             currency_df.to_csv(symbol + '__D.csv')
        #
        #     args = ((symbol, first_time_d[symbol]) for symbol in AvailableDayLastCandle)
        #     num_threads_for_d_last_candle = len(AvailableDayLastCandle)
        #     if num_threads_for_d_last_candle > 0:
        #         with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads_for_d_last_candle) as executor:
        #             symbols = executor.map(threader_d_last_candle_atr_sprtrend, args)
        #
        #         for symbol in symbols:
        #             first_time_d[symbol] = False
        #             with SymbolReadWriteLocks[symbol]:
        #                 currency_df = pd.read_csv(symbol + '__D.csv', index_col=0)
        #
        #             # day candle analyze
        #             if (currency_df.iloc[-1]['SelfNrATR'] > -1.0 or currency_df.iloc[-1]['score_short'] > 1000.0 or
        #                 currency_df.iloc[-1]['score_long'] > 1000.0) and (currency_df.iloc[-1]['BySlDfNowPerc'] >= 0.0):
        #                 if currency_df.iloc[-1]['Price_st'] == 3 or currency_df.iloc[-1]['Price_st'] == 4 or \
        #                         currency_df.iloc[-1]['Price_st'] == 5:
        #                     if (currency_df.iloc[-1]['ByPr_st'] >= 0 and currency_df.iloc[-1]['BySlDf_st'] >= 0) or (
        #                             currency_df.iloc[-1]['ByPr_st'] <= 0 and (
        #                             currency_df.iloc[-1]['BySlDf_st'] >= 2 or currency_df.iloc[-2]['BySlDf_st'] >= 2)):
        #                         DayBuyCandidates_sy.append(symbol)
        #                         DayBuyCandidates_sy_sc_pos.append(
        #                             [symbol, currency_df.iloc[-1]['score_short'], 'short'])
        #
        #             elif (Kl_1h_Pass and currency_df.iloc[-1]['score_short'] > 0.0 and currency_df.iloc[-1][
        #                 'score_long'] > 0.0) \
        #                     and (
        #                     currency_df.iloc[-1]['BySlDfNowPerc'] > 0.0 and currency_df.iloc[-1]['ClsSlp2kl'] > 0.0) \
        #                     and (currency_df.iloc[-1]['BySlDf_st'] >= 2 or currency_df.iloc[-2]['BySlDf_st'] >= 2) \
        #                     and (currency_df.iloc[-1]['ClsSlpSlp16kl4kl'] > 0.0 and currency_df.iloc[-1][
        #                 'ClsSlpSlp32kl4kl'] > 0.0) \
        #                     and ((currency_df.iloc[-4]['ClsSlpSlp16kl4kl'] < 0.0 or currency_df.iloc[-3][
        #                 'ClsSlpSlp16kl4kl'] < 0.0 or currency_df.iloc[-2]['ClsSlpSlp16kl4kl'] < 0.0)
        #                          or (currency_df.iloc[-4]['ClsSlpSlp32kl4kl'] < 0.0 or currency_df.iloc[-3][
        #                         'ClsSlpSlp32kl4kl'] < 0.0 or currency_df.iloc[-2]['ClsSlpSlp32kl4kl'] < 0.0)):
        #                 DayBuyCandidates_sy.append(symbol)
        #                 DayBuyCandidates_sy_sc_pos.append([symbol, currency_df.iloc[-1]['score_long'], 'long'])
        #
        #             with SymbolReadWriteLocks[symbol]:
        #                 # export DataFrame to csv
        #                 currency_df.to_csv(symbol + '__D.csv')
        #     else:
        #         print('poor connection, day last candle cannot be received !!')
        #
        #     print('DayBuyCandidates are : ' + str(DayBuyCandidates_sy_sc_pos))
        #
        #     for DayCandidate in DayBuyCandidates_sy_sc_pos:
        #         for HourCandidate in HourBuyCandidates_sy_sc_pos:
        #             if HourCandidate[0] == DayCandidate[0]:
        #                 FinalBuyCandidates_sy_scD_scH_pos.append(
        #                     [DayCandidate[0], DayCandidate[1], HourCandidate[1], HourCandidate[2]])
        #                 break
        #     print('FinalBuyCandidates are : ' + str(FinalBuyCandidates_sy_scD_scH_pos))
        #     for Candidate in FinalBuyCandidates_sy_scD_scH_pos:
        #         if (Candidate[1] > 1500) and (Candidate[2] > 1500):
        #             urgentCandidates_sy.append(Candidate[0])
        #             urgentCandidates_sy_pos.append([Candidate[0], Candidate[3]])
        #     # print( 'urgentCandidates are ' + str( urgentCandidates_sy_pos ) )
        #     # bestCandidate = max( HourBuyCandidates_sy_sc_pos, key = lambda x : x[1] )
        #     # winners2.append( bestCandidate[0] )
        #     # if len( HourBuyCandidates_sy_sc_pos ) > 1 :
        #     #     HourBuyCandidates_sy_sc_pos.remove( bestCandidate )
        #     #     bestCandidate = max( HourBuyCandidates_sy_sc_pos, key = lambda x : x[1] )
        #     #     winners2.append( bestCandidate[0] )
        #
        # minBalance = 0.02
        # topUp = 0.1
        # bnbOrder = topup_bnb(client, minBalance, topUp)
        #
        # for sy_pos in urgentCandidates_sy_pos:
        #     print('urgent: ' + str(sy_pos[0]))
        #     th_sell = threading.Thread(target=threader_sell_scenario,
        #                                args=(Binance_Time_Kl_Buffer_5m, max_num_active_investment, budget, sy_pos[0],
        #                                      sy_pos[1]),
        #                                daemon=True)  # classifying as a daemon, so they will die when the main dies
        #     th_sell.start()
        #
        # sySH = []
        # syLO = []
        # print('prev_potential_short are : ' + str(prev_potential_short))
        # print('prev_potential_long are : ' + str(prev_potential_long))
        #
        # for sy_scD_scH_pos in FinalBuyCandidates_sy_scD_scH_pos:
        #     if sy_scD_scH_pos[1] > 1000 and sy_scD_scH_pos[2] > 1000 and sy_scD_scH_pos[3] == 'short':
        #         sySH.append(sy_scD_scH_pos[0])
        #         for episode in prev_potential_short:
        #             if (sy_scD_scH_pos[0] in episode) and (sy_scD_scH_pos[0] not in urgentCandidates_sy):
        #                 print('high potential (short): ' + str(sy_scD_scH_pos[0]))
        #                 th_sell = threading.Thread(target=threader_sell_scenario,
        #                                            args=(Binance_Time_Kl_Buffer_5m, max_num_active_investment, budget,
        #                                                  sy_scD_scH_pos[0], sy_scD_scH_pos[3]),
        #                                            daemon=True)  # classifying as a daemon, so they will die when the main dies
        #                 th_sell.start()
        #                 break
        #
        #     elif sy_scD_scH_pos[3] == 'long':
        #         syLO.append(sy_scD_scH_pos[0])
        #         for episode in prev_potential_long:
        #             if (sy_scD_scH_pos[0] in episode) and (sy_scD_scH_pos[0] not in urgentCandidates_sy):
        #                 print('high potential (long): ' + str(sy_scD_scH_pos[0]))
        #                 th_sell = threading.Thread(target=threader_sell_scenario,
        #                                            args=(Binance_Time_Kl_Buffer_5m, max_num_active_investment, budget,
        #                                                  sy_scD_scH_pos[0], sy_scD_scH_pos[3]),
        #                                            daemon=True)  # classifying as a daemon, so they will die when the main dies
        #                 th_sell.start()
        #                 break
        #
        # prev_potential_short.append(sySH)
        # if len(prev_potential_short) > 20:
        #     prev_potential_short = prev_potential_short[1:]
        # if Kl_1h_Pass:
        #     prev_potential_long.append(syLO)
        #     if len(prev_potential_long) > 4:
        #         prev_potential_long = prev_potential_long[1:]
