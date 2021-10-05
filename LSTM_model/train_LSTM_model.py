import os
import sys
import numpy as np
import tensorflow as tf
from math import floor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

o_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(o_path)
import data_prepare


def train_lstm_model(n_steps, n_inputs, n_neurons, n_label, learning_rate, d_begin, d_end, frequency, train_stocks,
                     features, s_point, train_scale, batch_size, n_epoch, keep_in_prob, keep_out_prob, day_period):

    tf.reset_default_graph()
    x = tf.compat.v1.placeholder(tf.float32, [None, n_steps, n_inputs], name='x')  # 测试集size未知
    y = tf.compat.v1.placeholder(tf.int32, [None], name='y')
    in_prob = tf.compat.v1.placeholder(tf.float32, [], name='in_prob')
    out_prob = tf.compat.v1.placeholder(tf.float32, [], name='out_prob')

    # 定义网络结构
    lstm_layer_1 = tf.contrib.rnn.LSTMCell(num_units=n_neurons[0], use_peepholes=True, state_is_tuple=False)
    lstm_layer_2 = tf.contrib.rnn.LSTMCell(num_units=n_neurons[1], use_peepholes=True, state_is_tuple=False)
    layer_2_dropout = tf.contrib.rnn.DropoutWrapper(lstm_layer_2, input_keep_prob=out_prob, output_keep_prob=in_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_layer_1, layer_2_dropout], state_is_tuple=False)
    outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    logits = tf.layers.dense(outputs[:, -1], n_label)  # 这里的全连接层是没有激活函数的，相当于做了一个线性映射以改变输出维度

    # 定义损失函数和优化器
    x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(x_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    prediction = tf.nn.softmax(logits)
    correct = tf.nn.in_top_k(prediction, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    s_loss = tf.summary.scalar('loss', loss)
    train_accuracy = tf.summary.scalar('train_accuracy', accuracy)
    test_accuracy = tf.summary.scalar('test_accuracy', accuracy)
    merged = tf.summary.merge([s_loss, train_accuracy])

    np.set_printoptions(suppress=True)
    np.set_printoptions(threshold=np.inf)
    data_x, data_y = data_prepare.fetch_data(d_begin, d_end, frequency, train_stocks, features, s_point, day_period)
    # LSTM模型只需要三维数据，将日内time_steps与day_period两个维度合并
    data_x = data_x.reshape(data_x.shape[0], n_steps, n_inputs)
    data_x = data_prepare.data_normalize(data_x)
    train_x, train_y, test_x, test_y = data_prepare.data_seperate(data_x, data_y, train_scale)
    n_batch = floor(len(train_x) / batch_size)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init.run()
        str_time = data_prepare.get_str_time()
        log_path = 'final_log/' + str_time + '/'
        writer = tf.summary.FileWriter(log_path, sess.graph)
        for epoch in range(n_epoch):
            print('—— epoch:', epoch)
            train_x, train_y = data_prepare.fetch_train_batch(train_x, train_y, batch_size, n_batch, n_steps, n_inputs)
            for batch_index in range(n_batch):
                x_batch = train_x[batch_index]
                y_batch = train_y[batch_index]
                if batch_index % 250 == 0:
                    step = epoch * n_batch + batch_index
                    print('No.Step:', step)
                    summary = sess.run(merged, feed_dict={x: x_batch, y: y_batch, in_prob: keep_in_prob, out_prob: keep_out_prob})
                    summary_test = test_accuracy.eval(feed_dict={x: test_x, y: test_y, in_prob: 1.0, out_prob: 1.0})
                    writer.add_summary(summary_test, epoch)
                    writer.add_summary(summary, epoch)
                sess.run(training_op, feed_dict={x: x_batch, y: y_batch, in_prob: keep_in_prob, out_prob: keep_out_prob})

            if epoch == 19 or epoch == 29 or epoch == 39 or epoch == 49:
                # 在不过拟合的时候保存分类报告、绘制混淆矩阵热力图
                str_time = data_prepare.get_str_time()
                report_path = 'final_report/' + str_time + '.txt'
                with open(report_path, 'w', encoding='utf-8') as f:
                    pred_prob = np.array(prediction.eval(feed_dict={x: test_x, in_prob: 1.0, out_prob: 1.0}),
                                         dtype=np.float32)
                    pred_y = np.argmax(pred_prob, axis=1)
                    target_names = ['class 0', 'class 1', 'class 2', 'class 3']
                    report = classification_report(test_y, pred_y, target_names=target_names)
                    f.write(report)
                    cm = confusion_matrix(test_y, pred_y)
                    figure_path = 'final_figure/' + str_time
                    data_prepare.plot_confusion_matrix(cm, target_names, figure_path, 'index')
                    data_prepare.plot_confusion_matrix(cm, target_names, figure_path, 'columns')
                    f.write(str(cm))
                    model_path = 'final_model/' + str_time + '/final_model.ckpt'
                    saver.save(sess, model_path, global_step=epoch + 1)
        saver.save(sess, model_path, global_step=n_epoch + 1)
        writer.close()


'''
tensorboard --logdir=C:\Users\86186\PycharmProjects\HFStock-Public\LSTM_model\LSTM\log
http://localhost:6006/
'''
