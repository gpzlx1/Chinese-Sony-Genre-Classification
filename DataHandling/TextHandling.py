import sys
from imp import reload
import re
import os
import multiprocessing
from Cat_to_Chs import *

reload(sys)


def getlist():
    song_list = []
    work_dir = "data/"
    for parent, dirnames, filenames in os.walk(work_dir):
        for filename in filenames:
            file_path = os.path.join(parent,filename)
            # print(file_path)
            song_list.append(file_path)
            # list = file_path.split();
            # print(list)
    return song_list


# sys.setdefaultencoding('utf8')

def is_chinese(char):
    if char >= '\u4e00' and char <= '\u9fa5':
        return True
    else:
        return False


# def is_alphabet(char):
#     if (char >= '\u0041' and char <= '\u005a') or (char >= '\u0061' and char <= '\u007a'):
#         return True
#     else:
#         return False


def text_handling(file_name):
    file = open(file_name, mode='r')

    ## 文本过滤
    str1 = file.read()
    file.close()

    str1 = re.sub(r'[(（].*[)）]', '', str1)
    str1 = re.sub(r'[^\n].*[:：].*\n', '', str1)
    str1 = re.sub(r'[^\n].*[-—]+.*\n', '', str1)

    punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=\\：｜’‘；、。，？》《【】「」{}“”'
    str1 = re.sub(r'[%s]+' % punc, ' ', str1)
    str1 = re.sub(r' +', ' ', str1)
    return str1


def title_handling(file_name, str1):
    file_name = re.sub(r'.*/', '', file_name)
    if re.search(r'[《].*[》]', file_name):
        work_dir = 'data/data1/'
        file_name = '{}{}'.format(work_dir, file_name)
        file = open(file_name, mode='w')
        file.write(str1)
        file.close()
        return
    result = re.search(r'\d+_[\u4e00-\u9fa5]+', file_name)
    if result:
        # file_name = re.sub(r'[(（【\[].*[)）】\]]', '', file_name)
        # file_name = re.sub(r'[-—].*', '.txt', file_name)
        work_dir = 'data/data2/'
        file_name = '{}{}{}'.format(work_dir, result.group(0), '.txt')
        file = open(file_name, mode='w')
        file.write(str1)
        file.close()
    else:
        work_dir = 'data/data3/'
        file_name = '{}{}'.format(work_dir, file_name)
        file = open(file_name, mode='w')
        file.write(str1)
        file.close()


def chinese_or_not(str1):
    # print(str1)

    ## 中文歌判断
    str1 = re.sub(r'[ \n]', '', str1)
    # print(str1)
    if (len(str1)) <= 30:
        return False
    chinesecount = 0
    count = 0
    for i in range(len(str1)):
        # print(str1[i])
        if is_chinese(str1[i]):
            chinesecount = chinesecount + 1
        count = count + 1
    # print(chinesecount / count)
    if (chinesecount / count >= 0.6):
        return True
    else:
        return False


def data_handling(file_name):
    try:
        str1 = text_handling(file_name)
        if chinese_or_not(str1):
            str1 = traditional2simplified(str1)
            title_handling(file_name, str1)
        else:
            return
    except BaseException as e:
        print(file_name,e)
        return



def main():
    os.mkdir('data/data1')
    os.mkdir('data/data2')
    os.mkdir('data/data3')
    # multiporcessing
    song_list = getlist()
    with multiprocessing.Pool(8) as p:
        p.map(data_handling, song_list)


if __name__ == "__main__":
    main()
