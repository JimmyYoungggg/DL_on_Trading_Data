import pandas as pd

'''
获取符合格式要求的股票池的股票代码序列：
沪深300指数成分股和中证800指数成分股
'''


def str_insert(str1, char, index):
    x = list(str1)
    x.insert(index, char)
    str2 = "".join(x)
    return str2


def str_pop(str1, index=-1):
    x = list(str1)
    y = x.pop(index)
    str2 = "".join(x)
    return str2


def get_hs_300():
    data_df = pd.read_csv('../data/hs_300.csv', index_col=None, encoding='utf-8')
    code = list(data_df['沪深300成分股'])
    for i in range(300):
        code[i] = code[i].replace('.', '')
        if code[i][7] == 'Z':
            code[i] = str_insert(code[i], 'Z', 0)
            code[i] = str_insert(code[i], 'S', 0)
        if code[i][7] == 'H':
            code[i] = str_insert(code[i], 'H', 0)
            code[i] = str_insert(code[i], 'S', 0)
        code[i] = str_pop(code[i], -1)
        code[i] = str_pop(code[i], -1)
    return code


def get_zz_800():
    data_df = pd.read_csv('../data/zz_800.csv', index_col=None, encoding='utf-8')
    code = list(data_df['成分券代码Constituent Code'])
    for i in range(len(code)):
        code[i] = str(code[i])
        if len(code[i]) == 6:
            if code[i][0] == '6':
                code[i] = str_insert(code[i], 'SH', 0)
            elif code[i][0] == '3':
                code[i] = str_insert(code[i], 'SZ', 0)
        else:
            while len(code[i]) < 6:
                code[i] = str_insert(code[i], '0', 0)
            code[i] = str_insert(code[i], 'SZ', 0)
    return code



