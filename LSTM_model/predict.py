import tensorflow as tf

import config

s_n_steps = config.s_n_steps
l_n_steps = config.l_n_steps

# 关于运算图和网络结构的参数配置
n_inputs = config.n_inputs
n_neurons = config.n_neurons
n_label = config.n_label
train_stocks = config.train_stocks
s_point = config.s_point
average_point = config.average_point

# 训练出的模型位置,需要在训练完成后将模型手动移动到 long，short 文件夹内
model_path_long = './final_model/long/final_model.ckpt'
model_path_short = './final_model/short/final_model.ckpt'


# 日期包含begin不包含end；且data仅是一天的数据，即二维array；
def predict(data, n_steps):
    tf.reset_default_graph()
    if n_steps == l_n_steps:
        model_path = model_path_long
    else:
        model_path = model_path_short

    x = tf.compat.v1.placeholder(tf.float32, [None, n_steps, n_inputs], name='x')  # 测试集size未知
    in_prob = tf.compat.v1.placeholder(tf.float32, [], name='in_prob')
    out_prob = tf.compat.v1.placeholder(tf.float32, [], name='out_prob')

    lstm_layer_1 = tf.contrib.rnn.LSTMCell(num_units=n_neurons[0], use_peepholes=True, state_is_tuple=False)
    lstm_layer_2 = tf.contrib.rnn.LSTMCell(num_units=n_neurons[1], use_peepholes=True, state_is_tuple=False)
    layer_2_dropout = tf.contrib.rnn.DropoutWrapper(lstm_layer_2, input_keep_prob=out_prob, output_keep_prob=in_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_layer_1, layer_2_dropout], state_is_tuple=False)
    outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    logits = tf.layers.dense(outputs[:, -1], n_label)  # 这里的全连接层是没有激活函数的，相当于做了一个线性映射以改变输出维度
    prediction = tf.nn.softmax(logits)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init.run()
        saver.restore(sess, model_path)
        pred = prediction.eval(feed_dict={x: data, in_prob: 1.0, out_prob: 1.0})
        return pred  # 返回预测期望收益


# data_long,data_short与回测框架对接，是被预测日的大盘数据（格式同训练数据），stock_list是对应的股票名称，三者要对应上
def return_stocks_buy(data_long, data_short, stock_list):
    pred_long = predict(data_long, l_n_steps)
    pred_short = predict(data_short, s_n_steps)
    expectation = dict()
    for i, s in enumerate(stock_list):
        exp = 0
        for j in range(4):
            # 两个模型的预测结果各加权50%
            exp += average_point[str(j)] * pred_short[i][j] * 0.5 + average_point[str(j)] * pred_long[i][j] * 0.5
        expectation[s] = exp
    tmp = sorted(expectation.items(), reverse=True, key=lambda a: a[1])  # 排序
    stocks_to_buy = list()
    for i, t in enumerate(tmp):
        if float(t[1]) < 0.0014 or i >= 20:  # 考虑到手续费，仅在预期收益高于0.0014时买入，且不超过20只股票
            break
        else:
            stocks_to_buy.append(t[0])
    stocks_buy = stocks_to_buy
    print('stock_buy: ', stocks_buy)


'''
if __name__ == '__main__':
    # data_long,data_short与回测框架对接，是被预测日的大盘数据（格式同训练数据），stock_list是对应的股票代码，三者要对应上
    return_stocks_buy(data_long, data_short, stock_list)
'''