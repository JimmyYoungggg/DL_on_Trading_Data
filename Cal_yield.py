import pandas as pd
import copy

# 分别打开收盘价文件和开盘价文件
open_path = './data/open.csv'
close_path = './data/close.csv'

open_df = pd.read_csv(open_path, encoding='utf-8')
close_df = pd.read_csv(close_path, encoding='utf-8')

# 计算收盘价与开盘价之差
yield2_df = copy.deepcopy(close_df)
yield2_df.iloc[:, 1:] = (close_df.iloc[:, 1:]-open_df.iloc[:, 1:]) / open_df.iloc[:, 1:]

# 保存日收益率数据
yield2_df.to_csv('./data/yield_co.csv', encoding='utf-8')
