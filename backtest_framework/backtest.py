# -*- coding: utf-8 -*-
# coding: utf-8
# import outside modules
import datetime
import numpy as np
import pandas as pd
import backtest_evaluation
from collections import OrderedDict
import os

import time


class BackTest:
    def __init__(self, time_begin, time_end, initial_num, strategy_name, slippage_num=1, is_figure_report=True,
                 evaluation_fig_size=30, eda_list=None, is_twinx=False, is_interest=True,
                 is_candlestick=False, is_volume=False, factor_list=False):

        print('>> Initialize the back test system.')

        # basic strategy information
        self._time_begin = time_begin
        self._time_end = time_end
        self._initial_num = initial_num
        self._strategy_name = strategy_name

        # public variables
        self.num = self._initial_num
        self.strategy_performance = None
        self.stocks_buy = []

        # visualization parameters
        self.eda_list = eda_list
        self._is_figure_report = is_figure_report
        self._is_twinx = is_twinx
        self._is_interest = is_interest
        self._is_candlestick = is_candlestick
        self._is_volume = is_volume
        self.factor_list = factor_list
        self._evaluation_fig_size = evaluation_fig_size

        # time list
        self._time_list = self._time_list_load()

        # frequency list
        self._frequency_list = self._frequency_list_load()

        # specify path
        self._path = self._path_specify()

        # load transaction fee information
        self._transaction_fee_data = self._transaction_fee_data_load()

        # create trading log
        self._log_create()

        print('>> Initialization completes.')

    def __str__(self):
        return 'BackTest Object (name: %s)' % self._strategy_name

    __repr__ = __str__

    def run(self):
        while self.num <= len(self._time_list) - 3:
            # trading strategy
            self.strategy()

            # update log
            self._log_update()

            # next minute
            self.num += 1

        # factor_record
        if self.factor_list:
            factor_list_t = [list(row) for row in np.array(self.factor_list).T]
            stock_list = self._get_files(path='../data/' + '120m' + '/' + self._time_list[0], filetype='csv')
            factor_df = pd.DataFrame(factor_list_t, index=stock_list, columns=self._time_list[0:-2])
            factor_df.to_csv('../data/factor/' + self._strategy_name + '_' + self._time_begin + '-' + self._time_end + '.csv')

        else:
            # evaluate the performance of the strategy by analyzing the trading log
            self._evaluate()

    def strategy(self):
        # strategy needs to be implemented in the strategy layer
        pass

    def get_kline(self, frequency, days):
        # check if the ask is legal
        if frequency not in self._frequency_list:
            raise ValueError('bar is not formed for this frequency.')

        data = dict()

        for i in range(days):
            temp = self._data_load(data_time=self._time_list[self.num - i], frequency=frequency)
            if i == 0:
                data[str(i)] = temp
            else:
                data['-' + str(i)] = temp
        return data

    def _evaluate(self):
        # construct evaluator of the strategy
        path_information = [{'data_path': self._path['data_path'], 'log_path': self._path['log_path'],
                             'fig_path': self._path['fig_path'], 'rep_path': self._path['rep_path']}]
        evaluator = backtest_evaluation.BackTestEvaluation(path_information)

        # assess the strategy performance
        strategy_performance, _ = evaluator.assess(is_fig_report=self._is_figure_report,
                                                   fig_size=self._evaluation_fig_size,
                                                   eda_list=self.eda_list,
                                                   is_twinx=self._is_twinx,
                                                   is_interest=self._is_interest,
                                                   is_candlestick=self._is_candlestick,
                                                   is_volume=self._is_volume)

        # record the strategy performance
        self.strategy_performance = strategy_performance[0]

    def _data_load(self, data_time, frequency):
        # read stock names
        stocks = self._get_files(path='../data/' + frequency + '/' + data_time, filetype='csv')

        # read data
        data = dict()

        for i in stocks:
            data_df = pd.read_csv('../data/'+frequency + '/' + data_time + '/' + i + '.csv', index_col=False, encoding='ANSI')

            # convert pandas DataFrame into combination of dictionary and numpy array
            if data_df.empty:
                temp = dict()
                temp['empty'] = True
            else:
                temp = dict()
                temp['empty'] = False
                temp['时间'] = np.array(data_df['时间'].values, dtype=str)
                temp['收盘价'] = np.array(data_df['收盘价'].values, dtype=np.double)
                temp['开盘价'] = np.array(data_df['开盘价'].values, dtype=np.double)
                temp['最高价'] = np.array(data_df['最高价'].values, dtype=np.double)
                temp['最低价'] = np.array(data_df['最低价'].values, dtype=np.double)
                temp['成交量'] = np.array(data_df['成交量'].values, dtype=np.double)
                temp['成交金额'] = np.array(data_df['成交金额'].values, dtype=np.double)
                temp['成交笔数'] = np.array(data_df['成交笔数'].values, dtype=np.double)
                temp['上次价'] = np.array(data_df['上次价'].values, dtype=np.double)
                temp['委比'] = np.array(data_df['委比'].values, dtype=np.double)
                temp['量比'] = np.array(data_df['量比'].values, dtype=np.double)
                temp['买卖标识'] = np.array(data_df['买卖标识'].values, dtype=np.double)
                temp['主买量'] = np.array(data_df['主买量'].values, dtype=np.double)
                temp['主买金额'] = np.array(data_df['主买金额'].values, dtype=np.double)
                temp['主卖量'] = np.array(data_df['主卖量'].values, dtype=np.double)
                temp['主卖金额'] = np.array(data_df['主卖金额'].values, dtype=np.double)
                temp['委买'] = np.array(data_df['委买'].values, dtype=np.double)
                temp['委卖'] = np.array(data_df['委卖'].values, dtype=np.double)

            data[i] = temp

        return data

    def get_stock_list(self):
        stock_list = self._get_files(path='../data/' + '120m' + '/' + self._time_list[self.num], filetype='csv')
        return np.array(stock_list)

    def _time_list_load(self):
        time_list = self._get_dirs('../data/1m')

        # default time setting
        if (not self._time_begin) or (not self._time_end):
            if not self._time_begin:
                self._time_begin = time_list[0]
            if not self._time_end:
                self._time_end = time_list[-1]

        time_list = self._interval_select(time_list)

        return time_list

    def _frequency_list_load(self):
        frequency_list = self._get_dirs('../data')
        return frequency_list

    @staticmethod
    def _get_dirs(path):
        name = []
        for _, dirs, _ in os.walk(path):
            for i in dirs:
                name.append(i)
        return name

    @staticmethod
    def _get_files(path, filetype):
        name = []
        for _, _, files in os.walk(path):
            for i in files:
                if filetype in i:
                    name.append(i.replace('.csv', ''))
        return name

    @staticmethod
    def _transaction_fee_data_load():
        transaction_fee_data = dict()
        transaction_fee_data['手续费'] = 0.0002
        transaction_fee_data['印花税'] = 0.001
        return transaction_fee_data

    def _log_create(self):
        trade_log = open(file=self._path['log_path'], mode='w')

        # write title of the csv file
        trade_log.write('Time,')
        trade_log.write('Number,')
        trade_log.write('Stocks,')
        trade_log.write('Direction,')
        trade_log.write('Price,')
        trade_log.write('Profit Rate,')
        trade_log.write('Fee Rate,')
        trade_log.write('Slippage Rate,')
        trade_log.write('Net Profit Rate\n')

        trade_log.close()

    def _log_update(self):
        trade_log = open(file=self._path['log_path'], mode='a')
        buy_price = self._get_stocks_open_price(self.stocks_buy, self._time_list[self.num + 1])
        sell_price = self._get_stocks_open_price(self.stocks_buy, self._time_list[self.num + 2])

        # write the buying information
        trade_log.write(str(self._time_list[self.num + 1]) + ',')
        trade_log.write(str(self.num) + ',')
        trade_log.write(self._list_to_str(self.stocks_buy) + ',')
        trade_log.write('Buy' + ',')
        trade_log.write(self._list_to_str(buy_price) + ',')
        trade_log.write('0' + ',')
        trade_log.write(str(self._transaction_fee_data['手续费'] if self.stocks_buy != [] else 0) + ',')
        trade_log.write(str(self._cal_slippage_rate() if self.stocks_buy != [] else 0) + ',')
        trade_log.write(str(0 - self._transaction_fee_data['手续费'] - self._cal_slippage_rate() if self.stocks_buy != [] else 0) + '\n')

        # write the selling information
        trade_log.write(str(self._time_list[self.num + 2]) + ',')
        trade_log.write(str(self.num) + ',')
        trade_log.write(self._list_to_str(self.stocks_buy) + ',')
        trade_log.write('Sell' + ',')
        trade_log.write(self._list_to_str(sell_price) + ',')
        trade_log.write(str(self._cal_profit_rate(buy_price, sell_price)) + ',')
        trade_log.write(str(((self._transaction_fee_data['手续费'] + self._transaction_fee_data['印花税']) *
                            (1 + self._cal_profit_rate(buy_price, sell_price))) if self.stocks_buy != [] else 0) + ',')
        trade_log.write(str(self._cal_slippage_rate() if self.stocks_buy != [] else 0) + ',')
        trade_log.write(str(((1 - self._transaction_fee_data['手续费'] - self._transaction_fee_data['印花税'] -
                             self._cal_slippage_rate()) * (1 + self._cal_profit_rate(buy_price, sell_price)) - 1)
                            if self.stocks_buy != [] else 0) + '\n')
        trade_log.close()

    @staticmethod
    def _get_stocks_open_price(stocks, time):
        price = []
        for i in stocks:
            data_df = pd.read_csv('../data/'+'120m/' + time + '/' + i + '.csv', index_col=False, encoding='ANSI')
            if data_df.empty:
                price.append(0)
            else:
                price.append(data_df['开盘价'][0])
            del data_df
        return price

    @staticmethod
    def _list_to_str(mylist):
        string = ''
        for i in mylist:
            string = string + str(i) + ';'
        return string

    @staticmethod
    def _cal_profit_rate(buy_price, sell_price):
        profit_rate = 0

        if len(buy_price) == 0:
            return profit_rate
        else:
            for i in range(len(buy_price)):
                if sell_price[i] != 0 and buy_price[i] != 0:
                    profit_rate += (sell_price[i] - buy_price[i]) / buy_price[i]
            return profit_rate / len(buy_price)

    @staticmethod
    def _cal_slippage_rate():
        return 0

    def _interval_select(self, time_list):
        begin_index = time_list.index(self._time_begin)
        end_index = time_list.index(self._time_end)
        return time_list[begin_index:(end_index+1)]

    def _path_specify(self):
        # define the return value
        path_dict = dict()

        # define the source of back-test data
        path_dict['data_path'] = '../data/SH000001.csv'

        # define where to store the trading log
        path_dict['log_path'] = '../log/{0}_{1}_{2}_LOG.csv'.format(self._strategy_name, self._time_begin,
                                                                    self._time_end)

        # define where to store the search_log
        path_dict['fig_path'] = '../search_log/{0}_{1}_{2}_FIG.png'.format(self._strategy_name, self._time_begin,
                                                                       self._time_end)

        # define where to store the trading search_report
        path_dict['rep_path'] = '../search_report/{0}_{1}_{2}_REP.csv'.format(self._strategy_name, self._time_begin,
                                                                       self._time_end)

        return path_dict
