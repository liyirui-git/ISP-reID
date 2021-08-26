'''
Author: Li, Yirui
Date: 2021-08-26
Description: 
FilePath: /liyirui/PycharmProjects/ISP-reID/utils/progress_bar.py
'''
import sys

def progress_bar(portion, total, length=50):
    """
    total 总数据大小，portion 已经传送的数据大小
    :param portion: 已经接收的数据量
    :param total: 总数据量
    :param length: 进度条的长度
    :return: 接收数据完成，返回True
    """
    sys.stdout.write('\r')
    temp_str = '[%-' + str(length) + 's] %d/%d %.2f%%'
    count = int(portion * length / total - 1)
    sys.stdout.write((temp_str % (('-' * count + '>'), portion, total, portion / total * 100)))
    sys.stdout.flush()

    if portion >= total:
        sys.stdout.write('\n')
        return True