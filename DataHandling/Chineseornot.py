import sys
from imp import reload

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
    word1 = ':'
    word2 = '：'
    word3 = '版权'
    str1 = ''
    for line in file:
        if (not word1 in line) and (not word2 in line) and (not word3 in line):
            str1 = '{}{}'.format(str1, line)
    # str1 = file.read()
    file.close()
    file_name_1 = '{}{}{}'.format(work_dir, filename, '_1.txt')
    file_1 = open(file_name_1, mode='w')
    file_1.write(str1)
    file_1.close()
    # print(str1)
    str2 = str1.replace('\n', '')
    str3 = str2.replace(' ', '')
    if (len(str3)) <= 20:
        return False
    chinesecount = 0
    count = 0
    for i in range(len(str3)):
        # print(str3[i])
        if is_chinese(str3[i]):
            chinesecount = chinesecount + 1
        count = count + 1
    if (chinesecount/count >= 0.9):
        return True
    else:
        return False



