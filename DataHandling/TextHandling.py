import sys
import re
import os
import multiprocessing
from Cat_to_Chs import *


target_dir_path = "/mnt/c/Users/gpzlx1/Desktop/netease/data/rock-songs"
out_dir_parent = re.sub('data', 'preprocessed_data', target_dir_path)
if not os.path.exists(out_dir_parent) or not os.path.isdir(out_dir_parent):
    os.mkdir(out_dir_parent) 

out_dir1 = out_dir_parent + "/finished/"
out_dir2 = out_dir_parent + "/problem/"
if os.path.exists(out_dir1) and os.path.isdir(out_dir1):
    os.system("rm -r {}".format(out_dir1))
os.mkdir(out_dir1)
if  os.path.exists(out_dir2) and os.path.isdir(out_dir2):
    os.system("rm -r {}".format(out_dir2))
os.mkdir(out_dir2)


def getlist(dir_path):
    work_dir = os.path.abspath(dir_path)
    song_list = os.listdir(work_dir)
    song_list = [ os.path.join(work_dir, s) for s in song_list ]
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

    punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
    str1 = re.sub(r'[%s]+' % punc, ' ', str1)
    str1 = re.sub(r' +', ' ', str1)
    return str1


def title_handling(file_name, str1, out_dir1, out_dir2):
    file_name = re.sub(r'.*/', '', file_name)
    if re.search(r'[《].*[》]', file_name):
        file_name = '{}{}'.format(out_dir2, file_name)
        return file_name

    result = re.search(r'\d+_[\u4e00-\u9fa5]+', file_name)
    if result:
        # file_name = re.sub(r'[(（【\[].*[)）】\]]', '', file_name)
        # file_name = re.sub(r'[-—].*', '.txt', file_name)
        work_dir = 'data/data2/'
        file_name = '{}{}{}'.format(out_dir1, result.group(0), '.txt')
    else:
        work_dir = 'data/data3/'
        file_name = '{}{}'.format(out_dir2, file_name)
 
    return file_name


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
            new_file_name = title_handling(file_name, str1, out_dir1, out_dir2)
            with open(new_file_name, mode='w') as f:
                f.write(str1)
        else:
            return
    except BaseException as e:
        print(file_name,e)
        return



def main():
    song_list = getlist(target_dir_path)
    with multiprocessing.Pool(12) as p:
        p.map(data_handling, song_list)



if __name__ == "__main__":
    main()
