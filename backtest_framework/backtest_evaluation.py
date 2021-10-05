import re
import talib
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_finance as mpf
from collections import OrderedDict


class BackTestEvaluation:
    def __init__(self, path_information):
        """
        path_information example:
        [{'data_path': data_path, 'log_path': log_path, 'fig_path': fig_path, 'rep_path': rep_path}, {...}, ..., {...}]
        """
        # individual strategies' information
        self._strategy_path = path_information
        self._strategy_information = []
        self._strategy_performance = []
        self._strategy_log = []
        self._strategy_num = len(self._strategy_path)

        # portfolio information
        self._portfolio_path = dict()
        self._portfolio_information = dict()
        self._portfolio_performance = dict()

        # back-test data
        self._data = None

    def assess(self, is_fig_report=True, fig_size=30,
               eda_list=None, is_twinx=False, is_interest=True,
               is_candlestick=False, is_volume=False):
        # preparation
        self._parse_information()
        self._read_log()
        self._read_data()

        # analyze individual strategies
        self._calculate_strategy_performance()
        self._print_strategy_performance()

        if is_fig_report:
            self._write_strategy_report()
            self._plot_strategy_performance(fig_size, eda_list, is_twinx, is_interest, is_candlestick, is_volume)

        # portfolio management
        if self._strategy_num >= 2:
            self._calculate_portfolio_performance()
            self._print_portfolio_performance()

            if is_fig_report:
                self._write_portfolio_report()
                self._plot_portfolio_performance(fig_size)
        else:
            pass

        return self._strategy_performance, self._portfolio_performance

    def _parse_information(self):
        # define the parsing pattern
        pattern1 = re.compile(r'[/](.*)[.]', re.S)
        pattern2 = re.compile(r'[/](.*)', re.S)

        # parse each strategy's information
        for i in range(self._strategy_num):
            path = self._strategy_path[i]
            information_list = re.findall(pattern2, re.findall(pattern1, path['log_path'])[0])[0].split('_')
            information_dict = dict()
            information_dict['strategy_name'] = information_list[0]
            information_dict['time_begin'] = information_list[1]
            information_dict['time_end'] = information_list[2]
            self._strategy_information.append(information_dict)

        # check if the back-test times are the same
        time_begin = self._strategy_information[0]['time_begin']
        time_end = self._strategy_information[0]['time_end']
        for i in range(1, self._strategy_num):
            if self._strategy_information[i]['time_begin'] != time_begin:
                raise ValueError('time_begin do not match.')
            elif self._strategy_information[i]['time_end'] != time_end:
                raise ValueError('time_end do not match.')
            else:
                continue

        # check if the back-test data is the same
        data_path = self._strategy_path[0]['data_path']
        for i in range(1, self._strategy_num):
            if self._strategy_path[i]['data_path'] != data_path:
                raise ValueError('data do not match.')
            else:
                continue

        # define the portfolio's information and path
        self._portfolio_information['time_begin'] = time_begin
        self._portfolio_information['time_end'] = time_end

        self._portfolio_path['data_path'] = data_path
        self._portfolio_path['fig_path'] = '../search_log/portfolio_{0}_{1}_FIG.png'.format(time_begin, time_end)
        self._portfolio_path['rep_path'] = '../search_report/portfolio_{0}_{1}_REP.csv'.format(time_begin, time_end)

    def _read_log(self):
        for i in range(self._strategy_num):
            self._strategy_log.append(pd.read_csv(self._strategy_path[i]['log_path'], index_col=False))

    def _read_data(self):
        # load csv file from disk
        data = pd.read_csv(self._portfolio_path['data_path'], index_col=False)

        # slice the data by time begin and time end
        begin_index = data.index[data['time'] == self._portfolio_information['time_begin']].tolist()[0] + 1
        end_index = data.index[data['time'] == self._portfolio_information['time_end']].tolist()[0] + 1
        self._data = data.iloc[begin_index:end_index, :].reset_index(drop=True)

    def _calculate_strategy_performance(self):
        for i in range(self._strategy_num):
            # import log
            log_file = self._strategy_log[i]

            # end the evaluation procedure if we don't have enough transactions.
            if len(log_file) <= 1:
                raise ValueError('No Complete Trading is Conducted!')
            else:
                pass

            # calculate back-test statistics
            simple_interest_rate_array = BackTestEvaluation.accumulate_interest(log_file['Net Profit Rate'].values)
            simple_interest_rate_array = BackTestEvaluation.delete_same_num(simple_interest_rate_array)
            simple_interest_rate_array = BackTestEvaluation.fill_interest_gap(log_file,
                                                                              simple_interest_rate_array,
                                                                              self._data)
            transaction_number = BackTestEvaluation.cal_transaction_number(log_file)
            win_number = BackTestEvaluation.cal_win_number(log_file)

            initial_price = self._data['close'].values[0]
            final_price = self._data['close'].values[-1]
            base_interest_rate = (final_price - initial_price) / initial_price

            maximal_draw_down_rate = BackTestEvaluation.cal_maximal_draw_down(simple_interest_rate_array)

            profit_in_rate = sum(log_file['Profit Rate'].values)

            fee_cost_in_rate = sum(log_file['Fee Rate'].values)

            slippage_cost_in_rate = sum(log_file['Slippage Rate'].values)

            net_profit_in_rate = sum(log_file['Net Profit Rate'].values)

            # record the performance statistics
            performance_dict = dict()
            performance_dict['transaction_number'] = transaction_number
            performance_dict['win_rate'] = win_number / transaction_number
            performance_dict['simple_interest_rate_array'] = simple_interest_rate_array

            performance_dict['profit_in_rate'] = profit_in_rate

            performance_dict['fee_cost_in_rate'] = fee_cost_in_rate

            performance_dict['slippage_cost_in_rate'] = slippage_cost_in_rate

            performance_dict['net_profit_in_rate'] = net_profit_in_rate

            performance_dict['simple_interest_rate_baseline'] = base_interest_rate
            performance_dict['maximal_draw_down_rate'] = maximal_draw_down_rate

            performance_dict['comprehensive_performance_rate'] = net_profit_in_rate / maximal_draw_down_rate

            # record or print the performance depending on the mode
            self._strategy_performance.append(performance_dict)

    def _print_strategy_performance(self):
        for i in range(self._strategy_num):
            print('---------------------------------------')
            print('Strategy Name: %s'
                  % self._strategy_information[i]['strategy_name'])
            print('Time Begin: %s'
                  % self._strategy_information[i]['time_begin'])
            print('Time End: %s'
                  % self._strategy_information[i]['time_end'])
            print('Baseline Simple Interest Rate: %.4f'
                  % self._strategy_performance[i]['simple_interest_rate_baseline'])
            print('Transaction Number: %i'
                  % self._strategy_performance[i]['transaction_number'])
            print('Winning Rate: %.4f'
                  % self._strategy_performance[i]['win_rate'])
            print('Profit in Rate: %.4f'
                  % self._strategy_performance[i]['profit_in_rate'])

            print('Fee Cost in Rate: %.4f'
                  % self._strategy_performance[i]['fee_cost_in_rate'])

            print('Slippage Cost in Rate: %.4f'
                  % self._strategy_performance[i]['slippage_cost_in_rate'])

            print('Net Profit in Rate: %.4f'
                  % self._strategy_performance[i]['net_profit_in_rate'])

            print('Maximal Draw Down in Rate: %.4f'
                  % self._strategy_performance[i]['maximal_draw_down_rate'])

            print('Interest/MDD (Rate): %.4f'
                  % self._strategy_performance[i]['comprehensive_performance_rate'])

    def _write_strategy_report(self):
        for i in range(self._strategy_num):
            # define the return value
            month_info = OrderedDict()
            month_list = BackTestEvaluation.month_range(self._strategy_information[i]['time_begin'][:10],
                                                        self._strategy_information[i]['time_end'][:10])
            for month in month_list:
                month_info[month] = dict()
                month_info[month]['revenue_rate'] = 0
                month_info[month]['revenue_rmb'] = 0
                month_info[month]['transaction_number'] = 0

            # loop through the ith strategy's trading log to update month information
            log_file = self._strategy_log[i]
            for j in range(len(log_file)):
                current_month = str(log_file['Time'].values[j])[:7]
                month_info[current_month]['revenue_rate'] += log_file['Net Profit Rate'].values[j]
                month_info[current_month]['transaction_number'] += 0.5

            # write strategy search_report into csv file
            strategy_report = open(file=self._strategy_path[i]['rep_path'], mode='w')
            strategy_report.write('month,revenue_rate,revenue_rmb,transaction_number\n')
            for month in month_info:
                strategy_report.write(month + ','
                                      + str(month_info[month]['revenue_rate']) + ','
                                      + str(int(month_info[month]['transaction_number'])) + '\n')
            strategy_report.write('\n')

            strategy_report.write('Strategy Name: %s\n'
                                  % self._strategy_information[i]['strategy_name'])
            strategy_report.write('Time Begin: %s\n'
                                  % BackTestEvaluation.time2datetime(self._strategy_information[i]['time_begin']))
            strategy_report.write('Time End: %s\n'
                                  % BackTestEvaluation.time2datetime(self._strategy_information[i]['time_end']))
            strategy_report.write('Baseline Simple Interest Rate: %.4f\n'
                                  % self._strategy_performance[i]['simple_interest_rate_baseline'])
            strategy_report.write('Transaction Number: %i\n'
                                  % self._strategy_performance[i]['transaction_number'])
            strategy_report.write('Winning Rate: %.4f\n'
                                  % self._strategy_performance[i]['win_rate'])
            strategy_report.write('Profit in Rate: %.4f\n'
                                  % self._strategy_performance[i]['profit_in_rate'])

            strategy_report.write('Fee Cost in Rate: %.4f\n'
                                  % self._strategy_performance[i]['fee_cost_in_rate'])

            strategy_report.write('Slippage Cost in Rate: %.4f\n'
                                  % self._strategy_performance[i]['slippage_cost_in_rate'])

            strategy_report.write('Net Profit in Rate: %.4f\n'
                                  % self._strategy_performance[i]['net_profit_in_rate'])

            strategy_report.write('Maximal Draw Down in Rate: %.4f\n'
                                  % self._strategy_performance[i]['maximal_draw_down_rate'])

            strategy_report.write('Interest/MDD (Rate): %.4f\n'
                                  % self._strategy_performance[i]['comprehensive_performance_rate'])

            strategy_report.close()

    def _plot_strategy_performance(self, fig_size, eda_list, is_twinx, is_interest, is_candlestick, is_volume):
        for i in range(self._strategy_num):

            # get strategy information
            log_file = self._strategy_log[i]
            strategy_name = self._strategy_information[i]['strategy_name']
            simple_interest_rate_array = self._strategy_performance[i]['simple_interest_rate_array']
            fig_path = self._strategy_path[i]['fig_path']

            # plot search_log of price
            fig = plt.figure(figsize=(fig_size, 20)) if is_volume else plt.figure(figsize=(fig_size, 10))
            ax1 = fig.add_subplot(211) if is_volume else fig.add_subplot(111)
            if is_candlestick:
                quotes = [(i, self._data['open'].values[i], self._data['high'].values[i],
                           self._data['low'].values[i], self._data['close'].values[i])
                          for i in range(len(self._data['time']))]
                mpf.candlestick_ohlc(ax1, quotes, width=1, colorup='y', colordown='c')
            else:
                ax1.plot(self._data['close'].values, color='lightgrey', label='{0} Price'.format('上证指数'))
                ax1.legend(loc=1)

            # add title and label of the search_log
            ax1.set_xlabel(self._strategy_information[0]['time_begin'] + '  ---  ' +
                           self._strategy_information[0]['time_end'])
            ax1.set_title('{0} Strategy'.format(strategy_name))
            ax1.grid()

            # plot exploratory data analysis (EDA)
            # if eda_list:
            #     if is_twinx:
            #         ax2 = ax1.twinx()
            #         for eda in eda_list:
            #             ax2.plot(eda['value'], label=eda['label'], color=eda['color'])
            #         ax2.legend(loc=2)
            #     else:
            #         for eda in eda_list:
            #             ax1.plot(eda['value'], label=eda['label'], color=eda['color'])


            # plot simple interest
            if is_interest:
                ax3 = ax1.twinx()
                ax3.plot(simple_interest_rate_array, label='Simple Interest in Rate', color='r')
                ax3.legend(loc=3)

            # plot volume
            if is_volume:
                ax4 = fig.add_subplot(212)
                ax4.bar([i for i in range(len(self._data['volume'].values))],
                        self._data['volume'].values,
                        color=['red' if self._data['open'][i] <= self._data['close'][i] else 'green'
                               for i in range(len(self._data['volume'].values))],
                        label = 'Volume')
                ax4.plot(talib.EMA(np.array(self._data['volume'].values, dtype=np.double), timeperiod=15),
                         label='Volume EMA',
                         color='black')
                ax4.legend(loc=1)

            # save_pre and show
            plt.savefig(fig_path)
            plt.show()

    def _calculate_portfolio_performance(self):
        # form the simple interest rate array of portfolio
        simple_interest_rate_array = np.zeros(len(self._strategy_performance[0]['simple_interest_rate_array']))
        for i in range(self._strategy_num):
            simple_interest_rate_array += self._strategy_performance[i]['simple_interest_rate_array']
        simple_interest_rate_array /= self._strategy_num

        self._portfolio_performance['simple_interest_rate_array'] = simple_interest_rate_array

        # calculate back-test statistics of portfolio
        maximal_draw_down_rate = BackTestEvaluation.cal_maximal_draw_down(self._portfolio_performance
                                                                          ['simple_interest_rate_array'])
        simple_interest_rate = self._portfolio_performance['simple_interest_rate_array'][-1]

        # record
        self._portfolio_performance['simple_interest_rate'] = simple_interest_rate
        self._portfolio_performance['maximal_draw_down_rate'] = maximal_draw_down_rate
        self._portfolio_performance['comprehensive_performance_rate'] = simple_interest_rate / maximal_draw_down_rate

    def _print_portfolio_performance(self):
        print('---------------------------------------')
        print('Strategy Name: Portfolio')
        print('Time Begin: %s' % BackTestEvaluation.time2datetime(self._portfolio_information['time_begin']))
        print('Time End: %s' % BackTestEvaluation.time2datetime(self._portfolio_information['time_end']))
        print('Simple Interest Rate: %.4f' % self._portfolio_performance['simple_interest_rate'])
        print('Maximal Draw Down Rate: %.4f' % self._portfolio_performance['maximal_draw_down_rate'])
        print('Interest/MDD (Rate): %.4f' % self._portfolio_performance['comprehensive_performance_rate'])

    def _write_portfolio_report(self):
        # define the return value
        portfolio_month_info = OrderedDict()
        month_list = BackTestEvaluation.month_range(BackTestEvaluation.time2datetime(self._strategy_information[0]
                                                                                     ['time_begin'])[:10],
                                                    BackTestEvaluation.time2datetime(self._strategy_information[0]
                                                                                     ['time_end'])[:10])
        for month in month_list:
            portfolio_month_info[month] = dict()
            portfolio_month_info[month]['revenue_rate'] = 0
            portfolio_month_info[month]['revenue_rmb'] = 0
            portfolio_month_info[month]['transaction_number'] = 0

        # loop through all strategies' logs to update portfolio month information
        for i in range(self._strategy_num):
            log_file = self._strategy_log[i]
            for j in range(len(log_file)):
                current_month = str(log_file['Time'].values[j])[:6]
                current_month = current_month[:4] + '-' + current_month[4:]
                portfolio_month_info[current_month]['revenue_rate'] += log_file['Net Profit Rate'].values[j] \
                                                                  / self._strategy_num
                portfolio_month_info[current_month]['revenue_rmb'] += log_file['Net Profit(RMB)'].values[j]
                portfolio_month_info[current_month]['transaction_number'] += 0.5

        # write strategy_rbHot search_report
        portfolio_report = open(file=self._portfolio_path['rep_path'], mode='w')
        portfolio_report.write('month,revenue_rate,revenue_rmb,transaction_number\n')
        for month in portfolio_month_info:
            portfolio_report.write(month + ',' +
                                   str(portfolio_month_info[month]['revenue_rate']) + ',' +
                                   str(portfolio_month_info[month]['revenue_rmb']) + ',' +
                                   str(int(portfolio_month_info[month]['transaction_number'])) + '\n')
        portfolio_report.write('\n')

        portfolio_report.write('Strategy Name: Portfolio\n')
        portfolio_report.write('Time Begin: %s\n'
                               % BackTestEvaluation.time2datetime(self._portfolio_information['time_begin']))
        portfolio_report.write('Time End: %s\n'
                               % BackTestEvaluation.time2datetime(self._portfolio_information['time_end']))
        portfolio_report.write('Trading Instrument: %s\n' % self._portfolio_information['instrument'])
        portfolio_report.write('Simple Interest in RMB: %.2f\n' % self._portfolio_performance['simple_interest_rmb'])
        portfolio_report.write('Simple Interest Rate: %.4f\n' % self._portfolio_performance['simple_interest_rate'])
        portfolio_report.write('Maximal Draw Down in RMB: %.2f\n'
                               % self._portfolio_performance['maximal_draw_down_rmb'])
        portfolio_report.write('Maximal Draw Down Rate: %.4f\n'
                               % self._portfolio_performance['maximal_draw_down_rate'])
        portfolio_report.write('Interest/MDD (RMB): %.4f\n'
                               % self._portfolio_performance['comprehensive_performance_rmb'])
        portfolio_report.write('Interest/MDD (Rate): %.4f\n'
                               % self._portfolio_performance['comprehensive_performance_rate'])

        portfolio_report.close()

    def _plot_portfolio_performance(self, fig_size):
        if self._strategy_num >= 2:
            # plot close price of the instrument
            fig = plt.figure(figsize=(fig_size, 10))
            ax1 = fig.add_subplot(111)
            ax1.plot(self._data['close'].values,
                     label='{0} Close'.format(self._portfolio_information['instrument']),
                     color='lightgrey')
            ax1.set_xlabel(self._portfolio_information['time_begin'] + ' --- ' +
                           self._portfolio_information['time_end'])
            ax1.set_title('Portfolio Performance')
            ax1.legend(loc=1)
            ax1.grid()

            # plot the portfolio
            ax2 = ax1.twinx()
            ax2.plot(self._portfolio_performance['simple_interest_rate_array'],
                     label='Portfolio Revenue in Rate',
                     color='r')
            ax2.legend(loc=2)
            plt.savefig(self._portfolio_path['fig_path'], dpi=400)
            plt.show()
        else:
            raise ValueError

    @classmethod
    def month_range(cls, begin_date, end_date):
        month_set = set()
        for date in cls.date_range(begin_date, end_date):
            month_set.add(date[0:7])
        month_list = []
        for month in month_set:
            month_list.append(month)
        return sorted(month_list)

    @staticmethod
    def time2datetime(time):
        """
        time example: 20100416091600
        data_time example: '2017-10-27 00:00:00'
        """
        time = str(time)
        date_time = time[:4] + '-' + time[4:6] + '-' + time[6:8] + ' ' + time[8:10] + ':' + time[10:12] + ':' + time[12:14]
        return date_time

    @staticmethod
    def date_range(begin_date, end_date):
        dates = []
        dt = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
        date = begin_date[:]
        while date <= end_date:
            dates.append(date)
            dt = dt + datetime.timedelta(1)
            date = dt.strftime("%Y-%m-%d")
        return dates

    @staticmethod
    def accumulate_interest(sequence):
        accumulated_interest_array = np.array([sum(sequence[:i + 1]) for i in range(len(sequence))])
        return accumulated_interest_array

    @staticmethod
    def delete_same_num(sequence):
        new_seq = np.delete(sequence, list(range(0, len(sequence)-1, 2)))
        return new_seq

    @staticmethod
    def cal_transaction_number(log):
        return len(log) // 2

    @staticmethod
    def cal_win_number(log):
        net_profit_rate_array = np.array(log['Net Profit Rate'])
        win_num = 0
        i = 0
        while i + 1 <= len(net_profit_rate_array) - 1:
            if net_profit_rate_array[i] + net_profit_rate_array[i + 1] > 0:
                win_num += 1
            else:
                pass
            i += 2
        return win_num

    @staticmethod
    def cal_maximal_draw_down(simple_interest_array):
        index_right = np.argmax(np.maximum.accumulate(simple_interest_array) - simple_interest_array)
        index_left = np.argmax(simple_interest_array[:index_right])
        maximal_draw_down = simple_interest_array[index_left] - simple_interest_array[index_right]
        return maximal_draw_down

    @classmethod
    def fill_interest_gap(cls, log_file, simple_interest_array, data):
        enter_exit_flag = np.array(log_file['Number'])
        enter_exit_flag = cls.delete_same_num(enter_exit_flag)
        simple_interest_array_fixed = []
        for i in range(len(enter_exit_flag)):
            if i == 0:
                gap_number = enter_exit_flag[i]
                if gap_number != 0:
                    fill = [0] * gap_number
                    simple_interest_array_fixed.extend(fill)
            simple_interest_array_fixed.append(simple_interest_array[i])
            if i < len(enter_exit_flag) - 1:
                gap_number = enter_exit_flag[i + 1] - enter_exit_flag[i] - 1
            else:
                gap_number = len(data['close'].values) - enter_exit_flag[i] - 1
            if gap_number != 0:
                simple_fill = [simple_interest_array_fixed[-1]] * gap_number
                simple_interest_array_fixed.extend(simple_fill)
        simple_interest_array = np.array(simple_interest_array_fixed)
        return simple_interest_array
