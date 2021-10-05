import get_stock_code
import data_prepare

# 参数配置
frequency = 15                        #使用多少分钟尺度的数据
n_inputs = 11                         #使用了多少个原始量价指标
day_period = 5                        #模型的输入是往前回溯多少天的数据
time_steps = 240 // frequency
n_steps = time_steps * day_period
n_neurons = [250, 100]                #两个全连接层分别的神经元数目
n_filters=40                          #卷积层的卷积核数目
learning_rate = 0.001                 #优化器学习率
batch_size = 30                       #分批次训练的批次大小
n_label = 4                           #大涨、大跌、小涨、小跌
n_epoch = 100                         #训练的迭代次数
train_scale = 0.8                     #样本数据的训练集比例
dropout_rate=0.4                      #全连接层的dropout rate
# 注意开始和截止日期要求是交易日
d_begin = '2014-07-01'                #样本集的起止日期
d_end = '2018-12-28'
features = ['收盘价', '开盘价', '最高价', '最低价', '成交量', '成交金额', '成交笔数', '委比', '量比', '委买', '委卖']    #选用的11个原始指标
train_stocks = get_stock_code.get_hs_300()  # 选用沪深300指数成分股为股票池
s_point, average_point = data_prepare.labels_select(train_stocks)     #股票日收益率的三个四分位数

