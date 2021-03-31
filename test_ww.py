import akshare as ak
import pandas as pd
import numpy as np
import time
from datetime import datetime
import sys, os


def get_all_stock_code():
    """
    获取沪深所有股票代码
    """
    import baostock as bs

    #### 登陆系统 ####
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)

    #### 获取证券信息 ####
    rs = bs.query_all_stock(day="2021-02-10")
    print('query_all_stock respond error_code:' + rs.error_code)
    print('query_all_stock respond  error_msg:' + rs.error_msg)

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    #### 结果集输出到csv文件 ####
    result.to_csv("E:\\all_stock.csv", encoding="gbk", index=False)
    print(result)

    #### 登出系统 ####
    bs.logout()


def get_all_stock_data():
    """
    新浪采集每只股票最近几天的数据
    """

    save_root = 'E:/stocks/%s' % (datetime.now().strftime("%Y-%m-%d"))
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    df = pd.read_csv('E:\\all_stock.csv', encoding='gbk')
    df.head()

    import pdb
    pdb.set_trace()

    length = len(df.code)
    for i, code in enumerate(df.code):
        code = code.replace('.', '')
        print('%d/%d %s' % (i, length, code))
        save_filename = "%s/%s.csv" % (save_root, code)
        if os.path.exists(save_filename):
            continue
        stock_zh_a_minute_df = ak.stock_zh_a_minute(symbol=code, period='1')
        print(stock_zh_a_minute_df)
        stock_zh_a_minute_df.to_csv(save_filename, encoding="utf-8", index=False)
        time.sleep(np.random.randint(3))


def analysis_one_stock_backup(code):
    """
    根据微微的规则，对每只股票进行分析
    """
    save_root = 'E:/stocks/%s' % (datetime.now().strftime("%Y-%m-%d"))
    try:
        df = pd.read_csv("%s/%s.csv" % (save_root, code), encoding="utf-8", parse_dates=['day'])
    except:
        # print('wrong file')
        return False

    df.head()

    dates = df.day.dt.date.unique().tolist()
    # print('dates', dates)
    if len(dates) < 2:
        return False

    prev_day = dates[-1]
    begin = pd.to_datetime(prev_day.strftime('%Y-%m-%d') + ' 09:29:30')
    end = pd.to_datetime(prev_day.strftime('%Y-%m-%d') + ' 09:34:30')
    df1 = df[(df['day'] >= begin) & (df['day'] <= end)]
    if len(df1) == 0:
        return False
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
    if len(df2) == 0:
        return False

    # 规则1: 9:30~9:35，k线收阳，当天交易额大于前一天交易额2倍以上
    start = df2.open.tolist()[0]
    stop = df2.close.tolist()[-1]
    if stop < start:
        # print('rule1 failed.')
        return False

    avg_price = (df1.open + df1.high + df1.low + df1.close) / 4
    all_amount1 = avg_price * df1.volume
    all_amount1 = all_amount1.sum()  # 前一日5分钟内交易额
    avg_price = (df2.open + df2.high + df2.low + df2.close) / 4
    all_amount2 = avg_price * df2.volume
    all_amount2 = all_amount2.sum()  # 今日5分钟内交易额
    if all_amount2 < 2 * all_amount1:
        # print('rule1 failed.')
        return False

    # 规则2: 9:35~9:40，k线收阳
    begin = pd.to_datetime(datetime.now().strftime('%Y-%m-%d') + ' 09:34:30')
    end = pd.to_datetime(datetime.now().strftime('%Y-%m-%d') + ' 09:39:30')
    df2 = df[(df['day'] >= begin) & (df['day'] <= end)]
    if len(df2) == 0:
        # print('rule2 failed.')
        return False
    start = df2.open.tolist()[0]
    stop = df2.close.tolist()[-1]
    if stop < start:
        # print('rule2 failed.')
        return False

    # 规则3: 9:40~9:45，k线收阳，当天交易额大于前一天交易额2倍以上

    begin = pd.to_datetime(prev_day.strftime('%Y-%m-%d') + ' 09:39:30')
    end = pd.to_datetime(prev_day.strftime('%Y-%m-%d') + ' 09:44:30')
    df1 = df[(df['day'] >= begin) & (df['day'] <= end)]
    if len(df1) == 0:
        # print('rule3 failed.')
        return False
    begin = pd.to_datetime(datetime.now().strftime('%Y-%m-%d') + ' 09:39:30')
    end = pd.to_datetime(datetime.now().strftime('%Y-%m-%d') + ' 09:44:30')
    df2 = df[(df['day'] >= begin) & (df['day'] <= end)]
    if len(df2) == 0:
        # print('rule3 failed.')
        return False

    start = df2.open.tolist()[0]
    stop = df2.close.tolist()[-1]
    if stop < start:
        # print('rule3 failed.')
        return False

    avg_price = (df1.open + df1.high + df1.low + df1.close) / 4
    all_amount1 = avg_price * df1.volume
    all_amount1 = all_amount1.sum()  # 前一日5分钟内交易额
    avg_price = (df2.open + df2.high + df2.low + df2.close) / 4
    all_amount2 = avg_price * df2.volume
    all_amount2 = all_amount2.sum()  # 今日5分钟内交易额
    if all_amount2 < 2 * all_amount1:
        # print('rule3 failed.')
        return False

    # 以上条件全满足，返回True
    return True


def analysis_one_stock_2(code):
    """
    根据微微的规则，对每只股票进行分析
    """
    # save_root = 'E:/stocks/%s' % (datetime.now().strftime("%Y-%m-%d"))
    save_root = 'E:/stocks/YYYY-MM-DD/'
    try:
        df = pd.read_csv("%s/%s.csv" % (save_root, code), encoding="utf-8", parse_dates=['day'])
    except:
        # print('wrong file')
        return False

    df.head()

    dates = df.day.dt.date.unique().tolist()
    # print('dates', dates)
    if len(dates) < 2:
        return False

    prev_day = dates[-1]
    begin = pd.to_datetime(prev_day.strftime('%Y-%m-%d') + ' 09:29:30')
    end = pd.to_datetime(prev_day.strftime('%Y-%m-%d') + ' 09:34:30')
    df1 = df[(df['day'] >= begin) & (df['day'] <= end)]
    if len(df1) == 0:
        return False
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
    if len(df2) == 0:
        return False

    # 规则1: 9:30~9:35，k线收阳，当天交易额大于前一天交易额2倍以上
    start = df2.open.tolist()[0]
    stop = df2.close.tolist()[-1]
    if stop < start:
        # print('rule1 failed.')
        return False

    all_amount1 = df1.volume.sum()  # 前一日5分钟内交易额
    all_amount2 = df2.volume.sum()  # 今日5分钟内交易额
    if all_amount2 < 2 * all_amount1:
        # print('rule1 failed.')
        return False

    # 规则2: 9:35~9:40，k线收阳
    begin = pd.to_datetime(datetime.now().strftime('%Y-%m-%d') + ' 09:34:30')
    end = pd.to_datetime(datetime.now().strftime('%Y-%m-%d') + ' 09:39:30')
    df2 = df[(df['day'] >= begin) & (df['day'] <= end)]
    if len(df2) == 0:
        # print('rule2 failed.')
        return False
    start = df2.open.tolist()[0]
    stop = df2.close.tolist()[-1]
    if stop < start:
        # print('rule2 failed.')
        return False

    # 规则3: 9:40~9:45，k线收阳，当天交易额大于前一天交易额2倍以上

    begin = pd.to_datetime(prev_day.strftime('%Y-%m-%d') + ' 09:39:30')
    end = pd.to_datetime(prev_day.strftime('%Y-%m-%d') + ' 09:44:30')
    df1 = df[(df['day'] >= begin) & (df['day'] <= end)]
    if len(df1) == 0:
        # print('rule3 failed.')
        return False
    begin = pd.to_datetime(datetime.now().strftime('%Y-%m-%d') + ' 09:39:30')
    end = pd.to_datetime(datetime.now().strftime('%Y-%m-%d') + ' 09:44:30')
    df2 = df[(df['day'] >= begin) & (df['day'] <= end)]
    if len(df2) == 0:
        # print('rule3 failed.')
        return False

    start = df2.open.tolist()[0]
    stop = df2.close.tolist()[-1]
    if stop < start:
        # print('rule3 failed.')
        return False

    all_amount1 = df1.volume.sum()  # 前一日5分钟内交易额
    all_amount2 = df2.volume.sum()  # 今日5分钟内交易额
    if all_amount2 < 2 * all_amount1:
        # print('rule3 failed.')
        return False

    # 以上条件全满足，返回True
    return True


def analysis_one_stock(code):
    """
    根据微微的规则，对每只股票进行分析
    """
    # save_root = 'E:/stocks/%s' % (datetime.now().strftime("%Y-%m-%d"))
    save_root = 'E:/stocks/YYYY-MM-DD/'
    try:
        df = pd.read_csv("%s/%s.csv" % (save_root, code), encoding="utf-8", parse_dates=['day'])
    except:
        # print('wrong file')
        return False

    df.head()

    dates = df.day.dt.date.unique().tolist()
    # print('dates', dates)
    if len(dates) < 2:
        return False

    prev_day = dates[-2]
    begin = pd.to_datetime(prev_day.strftime('%Y-%m-%d') + ' 09:29:30')
    end = pd.to_datetime(prev_day.strftime('%Y-%m-%d') + ' 09:34:30')
    df1 = df[(df['day'] >= begin) & (df['day'] <= end)]
    if len(df1) == 0:
        return False
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
    if len(df2) == 0:
        return False

    # 规则1: 9:30~9:35，k线收阳，当天交易额大于前一天交易额2倍以上
    start = df2.open.tolist()[0]
    stop = df2.close.tolist()[-1]
    if stop < start:
        # print('rule1 failed.')
        return False

    all_amount1 = df1.volume.sum()  # 前一日5分钟内交易额
    all_amount2 = df2.volume.sum()  # 今日5分钟内交易额
    if all_amount2 < 2 * all_amount1:
        # print('rule1 failed.')
        return False

    # 规则2: 9:35~9:40，k线收阳
    begin = pd.to_datetime(datetime.now().strftime('%Y-%m-%d') + ' 09:34:30')
    end = pd.to_datetime(datetime.now().strftime('%Y-%m-%d') + ' 09:39:30')
    df2 = df[(df['day'] >= begin) & (df['day'] <= end)]
    if len(df2) == 0:
        # print('rule2 failed.')
        return False
    start = df2.open.tolist()[0]
    stop = df2.close.tolist()[-1]
    if stop < start:
        # print('rule2 failed.')
        return False

    # 规则3: 9:40~9:45，k线收阳，当天交易额大于前一天交易额2倍以上

    begin = pd.to_datetime(prev_day.strftime('%Y-%m-%d') + ' 09:39:30')
    end = pd.to_datetime(prev_day.strftime('%Y-%m-%d') + ' 09:44:30')
    df1 = df[(df['day'] >= begin) & (df['day'] <= end)]
    if len(df1) == 0:
        # print('rule3 failed.')
        return False
    begin = pd.to_datetime(datetime.now().strftime('%Y-%m-%d') + ' 09:39:30')
    end = pd.to_datetime(datetime.now().strftime('%Y-%m-%d') + ' 09:44:30')
    df2 = df[(df['day'] >= begin) & (df['day'] <= end)]
    if len(df2) == 0:
        # print('rule3 failed.')
        return False

    start = df2.open.tolist()[0]
    stop = df2.close.tolist()[-1]
    if stop < start:
        # print('rule3 failed.')
        return False

    all_amount1 = df1.volume.sum()  # 前一日5分钟内交易额
    all_amount2 = df2.volume.sum()  # 今日5分钟内交易额
    if all_amount2 < 2 * all_amount1:
        # print('rule3 failed.')
        return False

    # 以上条件全满足，返回True
    return True



def analysis_all():
    import multiprocessing
    from multiprocessing import Pool
    df = pd.read_csv('E:\\all_stock.csv', encoding='gbk')

    codes = df.code.tolist()
    codes = [code.replace('.', '') for code in codes]
    with Pool(multiprocessing.cpu_count()) as p:
        results = p.map(analysis_one_stock, codes)
    print(results)

    goods = []
    for i, x in enumerate(results):
        if x:
            goods.append(codes[i])

    print(len(goods))
    print(goods)

    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    get_all_stock_data()
    analysis_all()
