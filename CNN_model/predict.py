import os
import tensorflow as tf

import config


# 关于运算图和网络结构的参数配置
n_inputs = config.n_inputs
n_steps = config.n_steps
n_filters = config.n_filters
n_label = config.n_label
train_stocks = config.train_stocks
s_point = config.s_point
average_point = config.average_point
dropout_rate=config.dropout_rate
n_neurons=config.n_neurons


# 训练出的模型位置,需要在训练完成后将模型手动移动到 final 文件夹内
model_path = './final_model/model.ckpt'


# 日期包含begin 不包含end 且data是二维array
def predict(data, stock_list):
    tf.reset_default_graph()
    x = tf.compat.v1.placeholder(tf.float32, [None, n_steps, n_inputs], name='x')  # 测试集size未知
    tf_is_training = tf.compat.v1.placeholder(tf.bool, None)  # to control dropout when training and testing

    '''-------CNN-------'''
    input_layer = tf.reshape(x, [-1, n_steps, n_inputs, 1])  # train_x_batch之后再划分，n_steps:15m数据一天16条，现在有11个特征:n_inputs
    input_layer = tf.compat.v1.keras.layers.SpatialDropout2D(0.3)(inputs=input_layer, training=tf_is_training)
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=n_filters,  # 卷积核个数，为待定超参数
        kernel_size=[1, n_inputs],  # 提取高阶特征
        padding="valid",
        activation=tf.nn.relu)

    pool2_flat = tf.reshape(conv1, [-1, n_steps * n_filters])
    pool2_flat = tf.layers.dropout(pool2_flat, rate=dropout_rate, training=tf_is_training)
    dense1 = tf.layers.dense(inputs=pool2_flat, units=n_neurons[0], activation=tf.nn.relu)
    dense1 = tf.layers.dropout(dense1, rate=dropout_rate, training=tf_is_training)
    dense2 = tf.layers.dense(inputs=dense1, units=n_neurons[1], activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense2, units=n_label)

    prediction = tf.nn.softmax(logits)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init.run()
        saver.restore(sess, model_path)
        prediction_prob = prediction.eval(feed_dict={x: data, tf_is_training: False})
        expectation = dict()
        for i, s in enumerate(stock_list):
            exp = 0
            for j in range(4):
                exp += average_point[str(j)] * prediction_prob[i][j]
            expectation[s] = exp
        tmp = sorted(expectation.items(), reverse=True, key=lambda a: a[1])
        stocks_buy = list()
        for i, t in enumerate(tmp):
            if float(t[1]) < 0.0014 or i >= 10:
                break
            else:
                stocks_buy.append(t[0])
        return stocks_buy
