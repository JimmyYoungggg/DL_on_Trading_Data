import numpy as np
import tensorflow as tf
from math import floor
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


import data_prepare

'''-----file path where results are stored-----'''
str_time = data_prepare.get_str_time()
log_path = 'log/' + str_time + '/'
model_path = 'model/' + str_time + '/model.ckpt'
report_path = 'report/' + str_time + '.txt'
figure_path = 'figure/' + str_time

def plot_confusion_matrix(cm, labels_name, way_to_normalize='index'):
    if way_to_normalize == 'index':
        cm_float = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 行归一化（一般不会出现分母为0）但测试时要避免
    elif way_to_normalize == 'columns':
        cm_float = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]  # 列归一化
    plt.imshow(cm_float, interpolation='nearest')
    plt.title('Normalized by ' + way_to_normalize)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(x=i, y=j, s=int(cm[i][j]), va='center', ha='center', color='red', fontsize=15)
    plt.savefig(figure_path + '_' + way_to_normalize + '.png')
    plt.close()

'''-----MAIN function of CNN model and training-----'''
def train_cnn_model(n_steps, n_inputs,  n_label,n_neurons, learning_rate, d_begin, d_end, frequency, train_stocks,
                    features, s_point, train_scale, batch_size, n_epoch, n_filters,dropout_rate,day_period):

    x = tf.compat.v1.placeholder(tf.float32, [None, n_steps, n_inputs], name='x')
    y = tf.compat.v1.placeholder(tf.int32, [None], name='y')
    tf_is_training = tf.compat.v1.placeholder(tf.bool, None)  # to control dropout when training and testing

    input_layer = tf.reshape(x, [-1, n_steps, n_inputs, 1])
    input_layer = tf.compat.v1.keras.layers.SpatialDropout2D(0.3)(inputs=input_layer, training=tf_is_training)
    #为了与该模型的数据结构和dropout思路相吻合，需要SpatialDropout2D的原py文件中的line250中return里的
    # (input_shape[0], 1, 1, input_shape[3])改成(input_shape[0], input_shape[1], 1, input_shape[3])

    '''-----Model framework: CNN+2Dense-----'''
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=n_filters, 
        kernel_size=[1, n_inputs], 
        padding="valid",
        activation=tf.nn.relu)
    '''
    #The second convolution layer and pooling layer are abandoned because of bad experiment outcomes
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=n_filters,
        kernel_size=[3, 1],  
        padding="valid",
        activation=tf.nn.relu)
    pool1 = tf.layers.average_pooling2d(inputs=conv2, pool_size=[2, 1], strides=[2, 1])
    '''
    pool2_flat = tf.reshape(conv1, [-1, n_steps * n_filters])
    pool2_flat = tf.layers.dropout(pool2_flat, rate=dropout_rate, training=tf_is_training)
    dense1 = tf.layers.dense(inputs=pool2_flat, units=n_neurons[0], activation=tf.nn.relu)
    dense1 = tf.layers.dropout(dense1, rate=dropout_rate, training=tf_is_training)
    dense2 = tf.layers.dense(inputs=dense1, units=n_neurons[1], activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense2, units=n_label)  # 4分类问题：大涨，小涨，小跌，大跌

    '''-----Define loss function and optimizer----'''
    x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(x_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    '''-----Compute accuracy of prediction-----'''
    prediction = tf.nn.softmax(logits)
    correct = tf.nn.in_top_k(prediction, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    '''-----为了在tensorboard中画出准确率指标-----'''
    s_loss_train = tf.summary.scalar('loss_train', loss)
    s_loss_test = tf.summary.scalar('loss_test', loss)
    train_accuracy = tf.summary.scalar('train_accuracy', accuracy)
    test_accuracy = tf.summary.scalar('test_accuracy', accuracy)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    merged_train = tf.summary.merge([s_loss_train, train_accuracy])
    merged_test = tf.summary.merge([s_loss_test, test_accuracy])

    np.set_printoptions(suppress=True)
    np.set_printoptions(threshold=np.inf)


    '''-----Fetch data and devide them into training set and test set-----'''
    data_x, data_y = data_prepare.fetch_data(d_begin, d_end, frequency, train_stocks, features, s_point, day_period)
    data_x = data_x.reshape(data_x.shape[0], n_steps, n_inputs)
    data_x = data_prepare.data_normalize(data_x)
    train_x, train_y, test_x, test_y = data_prepare.data_seperate(data_x, data_y, train_scale)
    n_batch = floor(len(train_x) / batch_size)

    '''-----BEGIN training-----'''
    with tf.Session() as sess:
        init.run()
        writer = tf.summary.FileWriter(log_path, sess.graph)
        for epoch in range(n_epoch):
            print('—— epoch:', epoch)
            train_x_b, train_y_b = data_prepare.fetch_train_batch(train_x, train_y, batch_size, n_batch, n_steps,n_inputs)
            for batch_index in range(n_batch):
                x_batch = train_x_b[batch_index]
                y_batch = train_y_b[batch_index]
                sess.run(training_op, feed_dict={x: x_batch, y: y_batch, tf_is_training: True})
            summary_train = sess.run(merged_train, feed_dict={x: train_x, y: train_y, tf_is_training: False})
            summary_test = sess.run(merged_test, feed_dict={x: test_x, y: test_y, tf_is_training: False})
            writer.add_summary(summary_train, epoch)
            writer.add_summary(summary_test, epoch)
            summary_test_value = accuracy.eval(feed_dict={x: test_x, y: test_y, tf_is_training: False})
            print('accuracy of epoch No.', epoch, ':', summary_test_value)  #打印每个epoch结束后模型在验证集的正确率

        with open(report_path, 'w', encoding='utf-8') as f:  #保存分类报告
            pred_prob = np.array(prediction.eval(feed_dict={x: test_x, tf_is_training: False}),
                                dtype=np.float32)
            pred_y = np.argmax(pred_prob, axis=1)
            target_names = ['class 0', 'class 1', 'class 2', 'class 3']

            report = classification_report(test_y, pred_y, target_names=target_names)
            f.write(report)
            cm = confusion_matrix(test_y, pred_y)
            plot_confusion_matrix(cm, target_names, 'index')
            plot_confusion_matrix(cm, target_names, 'columns')
            f.write(str(cm))

        save_path = saver.save(sess, model_path)
        writer.close()

