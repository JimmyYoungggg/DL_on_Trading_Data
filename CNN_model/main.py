import numpy as np
import config
import train_CNN_model


# 参数配置
frequency = config.frequency
n_inputs = config.n_inputs
n_steps = config.n_steps
n_filters = config.n_filters
learning_rate = config.learning_rate
batch_size = config.batch_size
n_label = config.n_label
n_epoch = config.n_epoch
train_scale = config.train_scale
day_period = config.day_period
d_begin = config.d_begin
d_end = config.d_end
features = config.features
train_stocks = config.train_stocks
s_point = config.s_point
average_point = config.average_point
dropout_rate=config.dropout_rate
n_neurons=config.n_neurons

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)

'''-----训练CNN模型-----'''
train_CNN_model.train_cnn_model(n_steps, n_inputs, n_label, n_neurons, learning_rate, d_begin, d_end, frequency, train_stocks,
                                features, s_point, train_scale, batch_size, n_epoch, n_filters, dropout_rate, day_period)