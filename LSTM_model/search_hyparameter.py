import os
import sys
import tensorflow as tf
from math import floor
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

o_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(o_path)
import data_prepare

log_path = 'search_log/'
model_path = 'search_model/'
report_path = 'search_report/'
figure_path = 'search_figure/'


def get_str_hp(op_ch, re_ch, l2_ch, dr_ch):
    str_hp = op_ch + '+' + re_ch + '+'
    if re_ch == 'Dropout':
        str_hp = str_hp + dr_ch + '+'
    else:
        str_hp = str_hp + l2_ch + '+'
    str_hp += dr_ch
    return str_hp


def search_hyparameter(train_x, train_y, test_x, test_y, op_ch, re_ch, l2_ch, dr_ch, batch_size, n_steps, n_inputs, n_neurons,
                       n_label, learning_rate, n_epoch):

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
    cost = tf.reduce_mean(x_entropy)
    if re_ch == 'L2_regularization':
        tv = tf.trainable_variables()
        regularization_cost = float(l2_ch) * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
        loss = cost + regularization_cost
    else:
        loss = cost

    if op_ch == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif op_ch == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    else:
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    prediction = tf.nn.softmax(logits)
    correct = tf.nn.in_top_k(prediction, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # tensorboard 绘图
    s_loss = tf.summary.scalar('loss', loss)
    train_accuracy = tf.summary.scalar('train_accuracy', accuracy)
    test_accuracy = tf.summary.scalar('test_accuracy', accuracy)
    merged = tf.summary.merge([s_loss, train_accuracy])

    if re_ch == 'Dropout':
        if dr_ch == '1':
            i_p = 0.5
            o_p = 0.5
        elif dr_ch == '2':
            i_p = 0.6
            o_p = 0.6
        elif dr_ch == '3':
            i_p = 0.7
            o_p = 0.7
        elif dr_ch == '4':
            i_p = 0.8
            o_p = 0.8
        else:
            i_p = 0.9
            o_p = 0.9
    else:
        i_p = 1.0
        o_p = 1.0

    str_hp = get_str_hp(op_ch, re_ch, l2_ch, dr_ch)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_batch = floor(len(train_x) / batch_size)
    with tf.Session() as sess:
        init.run()
        writer = tf.summary.FileWriter(log_path + str_hp + '/', sess.graph)
        for epoch in range(n_epoch):
            print('search_model:' + str_hp + ' —— epoch:', epoch)
            train_x, train_y = data_prepare.fetch_train_batch(train_x, train_y, batch_size, n_batch, n_steps, n_inputs)
            for batch_index in range(n_batch):
                x_batch = train_x[batch_index]
                y_batch = train_y[batch_index]
                if batch_index % 80 == 0:
                    summary = sess.run(merged, feed_dict={x: x_batch, y: y_batch, in_prob: i_p, out_prob: o_p})
                    summary_test = test_accuracy.eval(feed_dict={x: test_x, y: test_y, in_prob: 1.0, out_prob: 1.0})
                    step = epoch * n_batch + batch_index
                    print('No.Step:', step)
                    writer.add_summary(summary, epoch)
                    writer.add_summary(summary_test, epoch)
                sess.run(training_op, feed_dict={x: x_batch, y: y_batch, in_prob: i_p, out_prob: o_p})

            if epoch == 29:
                # 在不过拟合的时候保存分类报告、绘制混淆矩阵热力图
                with open(report_path + str_hp + '.txt', 'w', encoding='utf-8') as f:
                    pred_prob = np.array(prediction.eval(feed_dict={x: test_x, in_prob: 1.0, out_prob: 1.0}), dtype=np.float32)
                    pred_y = np.argmax(pred_prob, axis=1)
                    target_names = ['class 0', 'class 1', 'class 2', 'class 3']
                    report = classification_report(test_y, pred_y, target_names=target_names)
                    f.write(report)
                    cm = confusion_matrix(test_y, pred_y)
                    data_prepare.plot_confusion_matrix(cm, target_names, figure_path + str_hp, 'index')
                    data_prepare.plot_confusion_matrix(cm, target_names, figure_path + str_hp, 'columns')
                    f.write(str(cm))
                    saver.save(sess, model_path + str_hp + '/search_model.ckpt', global_step=epoch + 1)
        saver.save(sess, model_path + str_hp + '/search_model.ckpt', global_step=n_epoch)
        writer.close()


'''
tensorboard --logdir=C:\Users\86186\PycharmProjects\HFStock-Public\LSTM_model\LSTM\log
http://localhost:6006/
'''
