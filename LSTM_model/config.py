import os
import sys

o_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(o_path)
import get_stock_code
import data_prepare

# 参数配置
# s(short),l(long)分别对应回溯1天，回溯10天的参数设置
s_frequency = 15
s_day_period = 1
s_time_steps = 240 // s_frequency
s_n_steps = s_time_steps * s_day_period

l_frequency = 120
l_day_period = 10
l_time_steps = 240 // l_frequency
l_n_steps = l_time_steps * l_day_period

n_inputs = 11
n_neurons = [100, 70]
learning_rate = 0.001
batch_size = 40
n_label = 4
n_epoch = 60
train_scale = 0.8
keep_in_prob = 0.8
keep_out_prob = 0.8
# 注意开始和截止日期要求是交易日
d_begin = '2014-07-01'
d_end = '2018-12-28'
features = ['收盘价', '开盘价', '最高价', '最低价', '成交量', '成交金额', '成交笔数', '委比', '量比', '委买', '委卖']
train_stocks = get_stock_code.get_hs_300()  # 选用沪深300指数成分股为股票池
s_point, average_point = data_prepare.labels_select(train_stocks)

