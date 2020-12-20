import sys
from imp import reload
import re

reload(sys)


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


def chineseornot(filename):
    work_dir = 'data/'
    file_name = '{}{}{}'.format(work_dir, filename, '.txt')
    file = open(file_name, mode='r')

    str1 = file.read()
    file.close()

    str1 = re.sub(r'[(（].*[)）]', '', str1)
    str1 = re.sub(r'[^\n].*[:：].*\n', '', str1)
    str1 = re.sub(r'[^\n].*[-—]+.*\n', '', str1)

    punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
    str1 = re.sub(r'[%s]+' %punc, ' ', str1)


    print(str1)
    str2 = str1.replace('\n', '')
    str3 = str2.replace(' ', '')
    if (len(str3)) <= 30:
        return False
    chinesecount = 0
    count = 0
    for i in range(len(str3)):
        print(str3[i])
        if is_chinese(str3[i]):
            chinesecount = chinesecount + 1
        count = count + 1
    print(chinesecount/count)
    if (chinesecount/count >= 0.6):
        return True
    else:
        return False


if __name__ == "__main__":
    print(chineseornot('1412478063'))
