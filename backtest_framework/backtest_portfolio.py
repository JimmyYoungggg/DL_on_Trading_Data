from backtest_evaluation import BackTestEvaluation


if __name__ == '__main__':
    path_information = [
                        {'data_path': '../data/SH000001.csv',
                         'log_path': '../log/T_2019-01-02_2019-01-15_LOG.csv',
                         'fig_path': '../search_log/T_2019-01-02_2019-01-15_FIG.png',
                         'rep_path': '../search_report/T_2019-01-02_2019-01-15_REP.csv'}
                        ]
    pm = BackTestEvaluation(path_information)
    pm.assess()
