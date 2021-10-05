# -*- coding: utf-8 -*-
# coding: utf-8
import backtest
import numpy as np
import time


class Template(backtest.BackTest):
    def __init__(self):
        # general information
        time_begin = '2019-01-02'
        time_end = '2019-05-31'
        initial_num = 0
        strategy_name = 'factor#001'
        slippage_num = 1.0

        # stop parameters

        # parameters

        # variables
        self.stocks_buy = []

        # visualization
        is_figure_report = True
        evaluation_fig_size = 30
        eda_list = None
        factor_list = None
        is_twinx = False
        is_interest = True
        is_candlestick = False
        is_volume = False

        super().__init__(time_begin, time_end, initial_num, strategy_name, slippage_num, is_figure_report,
                        evaluation_fig_size, eda_list, is_twinx, is_interest,
                        is_candlestick, is_volume, factor_list)

    def strategy(self):
        print(str(self.num))
        data = self.get_kline('120m', 2)
        self.stocks_buy = ['SH600000']


if __name__ == '__main__':
    strategy = Template()
    strategy.run()

