import os
import sys
import numpy as np

o_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(o_path)
import config
import data_prepare
import search_hyparameter
import train_LSTM_model

# 参数配置
s_frequency = config.s_frequency
s_day_period = config.s_day_period
s_n_steps = config.s_n_steps

l_frequency = config.l_frequency
l_day_period = config.l_day_period
l_n_steps = config.l_n_steps

n_inputs = config.n_inputs
n_neurons = config.n_neurons
learning_rate = config.learning_rate
batch_size = config.batch_size
n_label = config.n_label
n_epoch = config.n_epoch
train_scale = config.train_scale
keep_in_prob = config.keep_in_prob
keep_out_prob = config.keep_out_prob
d_begin = config.d_begin
d_end = config.d_end
features = config.features
train_stocks = config.train_stocks
s_point = config.s_point
average_point = config.average_point

# 网格搜索超参数
optimizer_grid = ['Adam', 'RMSProp', 'Adadelta']
regularization_grid = ['Dropout', 'L2_regularization']
L2_grid = ['0.001', '0.0005', '0.0002']
drop_grid = ['1', '2', '3', '4', '5']
# 1: in0.5*out0.5——2: in0.6*out0.6——3: in0.7*out0.7——4: in0.8*out0.8——5：in0.9*out0.9
batch_grid = ['20', '40']

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)

# 网格搜索超参数
data_x, data_y = data_prepare.fetch_data('2019-01-02', '2019-05-31', s_frequency, train_stocks, features, s_point, s_day_period)
# LSTM模型只需要三维数据，将日内time_steps与day_period两个维度合并
data_x = data_x.reshape(data_x.shape[0], s_n_steps, n_inputs)
data_x = data_prepare.data_normalize(data_x)
train_x, train_y, test_x, test_y = data_prepare.data_seperate(data_x, data_y, train_scale)

for op_ch in optimizer_grid:
    search_hyparameter.search_hyparameter(train_x, train_y, test_x, test_y, op_ch, 'Dropout', 'None', '2',
                                          batch_size, s_n_steps, n_inputs, n_neurons, n_label, learning_rate,
                                          n_epoch)
for l2_ch in L2_grid:
    search_hyparameter.search_hyparameter(train_x, train_y, test_x, test_y, 'Adam', 'L2_regularization', l2_ch, 'None',
                                          batch_size, s_n_steps, n_inputs, n_neurons, n_label, learning_rate,
                                          n_epoch)
for dr_ch in drop_grid:
    search_hyparameter.search_hyparameter(train_x, train_y, test_x, test_y, 'Adam', 'Dropout', 'None', dr_ch,
                                          batch_size, s_n_steps, n_inputs, n_neurons, n_label, learning_rate,
                                          n_epoch)

# 训练最终模型
train_LSTM_model.train_lstm_model(s_n_steps, n_inputs, n_neurons, n_label, learning_rate, d_begin, d_end, s_frequency,
                                  train_stocks, features, s_point, train_scale, batch_size, n_epoch,
                                  keep_in_prob, keep_out_prob, s_day_period)
train_LSTM_model.train_lstm_model(l_n_steps, n_inputs, n_neurons, n_label, learning_rate, d_begin, d_end, l_frequency,
                                  train_stocks, features, s_point, train_scale, batch_size, n_epoch,
                                  keep_in_prob, keep_out_prob, l_day_period)



