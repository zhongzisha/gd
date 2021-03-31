
import akshare as ak
import pandas as pd
import numpy as np
import time
from datetime import datetime
import sys, os


# def analysis_one_stock(code):
if __name__ == '__main__':
    code = 'sz002852'
    """
    根据微微的规则，对每只股票进行分析
    """
    # save_root = 'E:/stocks/%s' % (datetime.now().strftime("%Y-%m-%d"))
    save_root = 'E:/stocks/YYYY-MM-DD/'
    try:
        df = pd.read_csv("%s/%s.csv" % (save_root, code), encoding="utf-8", parse_dates=['day'])
    except:
        print('wrong file')
        # return False

    df.head()

    dates = df.day.dt.date.unique().tolist()
    print('dates', dates)
    # if len(dates) < 2:
    #     return False

    prev_day = dates[-2]
    begin = pd.to_datetime(prev_day.strftime('%Y-%m-%d') + ' 09:29:30')
    end = pd.to_datetime(prev_day.strftime('%Y-%m-%d') + ' 09:34:30')
    df1 = df[(df['day'] >= begin) & (df['day'] <= end)]

    """
                        day      open      high       low     close     volume
762 2021-02-18 09:31:00  6117.384  6129.054  6117.384  6125.741  443294100
763 2021-02-18 09:32:00  6126.910  6130.265  6123.140  6123.140  156522200
764 2021-02-18 09:33:00  6121.031  6121.514  6117.409  6117.409  154744000
765 2021-02-18 09:34:00  6116.220  6116.220  6090.058  6090.058  136061300
766 2021-02-18 09:35:00  6089.600  6089.600  6076.383  6076.383  124646800
    """

    begin = pd.to_datetime(datetime.now().strftime('%Y-%m-%d') + ' 09:29:30')
    end = pd.to_datetime(datetime.now().strftime('%Y-%m-%d') + ' 09:34:30')
    df2 = df[(df['day'] >= begin) & (df['day'] <= end)]

    # 规则1: 9:30~9:35，k线收阳，当天交易额大于前一天交易额2倍以上
    start = df2.open.tolist()[0]
    stop = df2.close.tolist()[-1]
    if stop < start:
        print('rule11 failed.')

    all_amount1 = df1.volume
    all_amount1 = all_amount1.sum()  # 前一日5分钟内交易额
    all_amount2 = df2.volume
    all_amount2 = all_amount2.sum()  # 今日5分钟内交易额
    print('1', all_amount1)
    print('2', all_amount2)
    if all_amount2 < 2 * all_amount1:
        print('rule12 failed.')

    # 规则2: 9:35~9:40，k线收阳
    begin = pd.to_datetime(datetime.now().strftime('%Y-%m-%d') + ' 09:34:30')
    end = pd.to_datetime(datetime.now().strftime('%Y-%m-%d') + ' 09:39:30')
    df2 = df[(df['day'] >= begin) & (df['day'] <= end)]
    if len(df2) == 0:
        print('rule21 failed.')

    start = df2.open.tolist()[0]
    stop = df2.close.tolist()[-1]
    if stop < start:
        print('rule22 failed.')


    # 规则3: 9:40~9:45，k线收阳，当天交易额大于前一天交易额2倍以上

    begin = pd.to_datetime(prev_day.strftime('%Y-%m-%d') + ' 09:39:30')
    end = pd.to_datetime(prev_day.strftime('%Y-%m-%d') + ' 09:44:30')
    df1 = df[(df['day'] >= begin) & (df['day'] <= end)]

    begin = pd.to_datetime(datetime.now().strftime('%Y-%m-%d') + ' 09:39:30')
    end = pd.to_datetime(datetime.now().strftime('%Y-%m-%d') + ' 09:44:30')
    df2 = df[(df['day'] >= begin) & (df['day'] <= end)]

    start = df2.open.tolist()[0]
    stop = df2.close.tolist()[-1]
    if stop < start:
        print('rule31 failed.')

    all_amount1 = df1.volume
    all_amount1 = all_amount1.sum()  # 前一日5分钟内交易额
    all_amount2 = df2.volume
    all_amount2 = all_amount2.sum()  # 今日5分钟内交易额
    if all_amount2 < 2 * all_amount1:
        print('rule32 failed.')


    # 以上条件全满足，返回True
    print('yes')
