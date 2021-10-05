import numpy as np
import os
import pandas as pd
from math import floor
import datetime
import matplotlib.pyplot as plt


# 指定一只股票的指定时间段的数据（list
def data_x_prepare(stock, dates, day_period, frequency, flag_df, features):
    tmp = list()  # tmp存序列
    for date in range(len(dates)):
        data_df = pd.read_csv('../data/' + str(frequency) + 'm/' + dates[date] + '/' + stock + '.csv', index_col=False,
                              encoding='ANSI')
        if data_df.empty:
            flag_df.loc[stock][date] = 0  # 如果当日为空，对当日做标记
        else:
            for feature in data_df.columns.values:
                if feature not in features:
                    del data_df[feature]
        tmp_list = data_df.values.tolist()
        tmp.append(tmp_list)  # 数据按时间轴方向，如果此日停牌还是会保留一个空的list作占位符
    data = [tmp[date-day_period: date] for date in range(day_period, len(dates))
            if flag_df.loc[stock][dates[date-day_period: date + 1]].min() == 1]  # 加一避免预测日停牌
    return data  # 四维数据：date(sample) * period * timestep * feature


# 获得y的list
def data_y_prepare(dates, flag_df, train_stocks, day_period):
    df = pd.read_csv('../data/yield_co.csv', encoding='ANSI')
    df = df.iloc[:, 1:]
    df.index = df["code"]
    del df["code"]
    for str_1 in df.columns:  # 将日期格式统一
        str_2 = str_1.replace('/', '-')
        flag = 0
        str_2 = list(str_2)
        for k in range(len(str_2) - 1, -1, -1):  # 倒序
            if str_2[k] != '-':
                flag += 1
            elif str_2[k] == '-' and flag == 1:
                str_2.insert(k + 1, '0')
                flag = 0
            elif str_2[k] == '-' and flag != 1:
                flag = 0
        str_2 = "".join(str_2)
        df = df.rename(columns={str_1: str_2})
    # 剔除不合规数据
    data_y = [df.loc[j][dates[i + day_period]] for j in train_stocks for i in range(len(dates) - day_period)
              if flag_df.loc[j][dates[i: i + day_period + 1]].min() == 1]
    return data_y  # 一维数据：(stock + date)连接


# 将y值离散化（转化为label）
def y_label(data, s_point):
    data_1 = list()
    for i in range(len(data)):
        if data[i] >= s_point['75%']:
            data_1.append(3)
        elif data[i] >= s_point['50%']:
            data_1.append(2)
        elif data[i] >= s_point['25%']:
            data_1.append(1)
        else:
            data_1.append(0)
    return data_1


# 获得array,x无论回溯一天还是多天统一为四维数据
def fetch_data(d_begin, d_end, frequency, train_stocks, features, s_point, day_period):
    dates = get_dirs(path='../data/' + str(frequency) + 'm', date_begin=d_begin, date_end=d_end)
    # 用来记录停牌的股票和日期，回溯的day_period天若出现停牌视为数据不合规，不作为输入样例，剔除
    flag_df = pd.DataFrame(1, index=train_stocks, columns=dates)
    data_x = list()
    for stock in train_stocks:
        data_x += data_x_prepare(stock, dates, day_period, frequency, flag_df, features)
        # 返回值可能为[]，一层的空list并不影响连接
        # stock 和 date 不区分，合并为一维, 按stock->date顺序展开
    data_y = data_y_prepare(dates, flag_df, train_stocks, day_period)
    data_y = y_label(data_y, s_point)  # 打标签
    data_x = np.array(data_x, dtype=np.float32)
    data_y = np.array(data_y, dtype=np.int32)
    return data_x, data_y


# 分为训练集和测试集
def data_seperate(data_x, data_y, train_scale):
    state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(state)
    np.random.shuffle(data_y)
    len_train = floor(len(data_x) * train_scale)
    train_x = data_x[: len_train]
    train_y = data_y[: len_train]
    test_x = data_x[len_train:]
    test_y = data_y[len_train:]
    return train_x, train_y, test_x, test_y


# 将数据随机分为batch
def fetch_train_batch(train_x, train_y, batch_size, n_batch, n_steps, n_inputs):
    state = np.random.get_state()
    np.random.shuffle(train_x)
    np.random.set_state(state)
    np.random.shuffle(train_y)
    train_x = train_x[0: n_batch * batch_size: 1].reshape(n_batch, batch_size, n_steps, n_inputs)
    train_y = train_y[0: n_batch * batch_size: 1].reshape(n_batch, batch_size)
    return train_x, train_y


# 数据归一化
def data_normalize(data):
    data = data.transpose((0, 2, 1))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            d_max = np.amax(data[i][j])
            d_min = np.amin(data[i][j])
            if d_max == d_min:
                data[i][j] = 0
            else:
                data[i][j] = np.divide(np.subtract(data[i][j], d_min), np.subtract(d_max, d_min))
    data = data.transpose((0, 2, 1))
    return data


# 获取目录——开始日和截止日之间的交易日日期
def get_dirs(path, date_begin, date_end):
    name = []
    for _, dirs, _ in os.walk(path):
        flag = False
        for i in dirs:
            if (flag == False) & (i == date_begin):
                name.append(i)
                flag = True
            elif (flag == True) & (i != date_end):
                name.append(i)
            elif i == date_end:
                name.append(i)
                break
    return name


# 取全部日收益率的四分位数为分类分界（四类），再求每一类所有收益率的平均值作为对应类别的收益率期望值
def labels_select(train_stocks):
    df = pd.read_csv('../data/yield_co.csv', encoding='utf-8')
    df = df.iloc[:, 1:]
    df.index = df["code"]
    del df["code"]
    data_1 = list()
    for stock in train_stocks:
        for j in df.columns.values:
            x = df.loc[stock][j]
            if np.isnan(x) == False and x != float('inf'):  # 处理空值（停牌等情况）
                data_1.append(x)
    data_2 = np.array(data_1, dtype=np.float32)
    s_point = dict()
    # 计算四分位数
    s_point['25%'] = np.percentile(data_2, 25)
    s_point['50%'] = np.median(data_2)
    s_point['75%'] = np.percentile(data_2, 75)
    average_point = dict()
    labels_num = {'0': 0, '1': 0, '2': 0, '3': 0}
    labels_sum = {'0': 0, '1': 0, '2': 0, '3': 0}
    # 累加得频率
    for i in data_2:
        if i < s_point['25%']:
            labels_num['0'] += 1
            labels_sum['0'] += i
        elif i < s_point['50%']:
            labels_num['1'] += 1
            labels_sum['1'] += i
        if i < s_point['75%']:
            labels_num['2'] += 1
            labels_sum['2'] += i
        else:
            labels_num['3'] += 1
            labels_sum['3'] += i
    # 计算收益率期望值
    for i in range(4):
        s = str(i)
        average_point[s] = labels_sum[s] / float(labels_num[s])
    return s_point, average_point


# 获取格式为 月-日_时-分-秒 的当前时间
def get_str_time():
    time = datetime.datetime.now()
    str_time = (str(time))[5:19]
    str_time = str_time.replace(' ', '_')
    str_time = str_time.replace(':', '-')  # 文件夹名称不能有冒号
    return str_time


# 绘制混淆矩阵热图
def plot_confusion_matrix(cm, labels_name, path, way_to_normalize='index'):
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
    plt.savefig(path + '_' + way_to_normalize + '.jpg')  # w+b 没办法创建新文件夹？
    plt.close()

