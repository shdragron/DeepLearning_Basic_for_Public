# 5_2_my_letters.py

import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image
import collections

# 숫자 0~9까지
# 소문자 a~z
# 대문자 A~Z
# 개당 10개

# applekoong@naver.com

# mgy_digit
# mgy_lower
# mgy_upper

# 숫자 0
# mgy_digit_0_0~mgy_digit_0_9

# 소문자 a
# mgy_a_0~mgy_a_9

# letters +- digits +- 0
import string
def make_letter_folder():
    for ch in list('0123456789'):
        os.makedirs(os.path.join('letters/digit/test', ch))
        os.makedirs(os.path.join('letters/digit/train', ch))
    for ch in list(string.ascii_uppercase):
        os.makedirs(os.path.join('letters/upper/test', ch))
        os.makedirs(os.path.join('letters/upper/train', ch))
    for ch in list(string.ascii_lowercase):
        os.makedirs(os.path.join('letters/lower/test', ch))
        os.makedirs(os.path.join('letters/lower/train', ch))


def copy_letters():
    freq, total = {}, []
    TRAIN_SIZE = 40
    for name in os.listdir('my_letters'):
        filename, ext = name.lower().split('.')
        i0, i1, i2, i3 = filename.split('_')
        # print(i0, i1, i2, i3)
        # total.append(i1+i2) # 'digit3', 'lowerp'

        key = i1 + i2
        if key not in freq:
            freq[key] = 0

        freq[key] += 1
        target = 'train' if freq[key] < TRAIN_SIZE else 'test'

        try:
            dst = os.path.join('letters/{}/{}/{}'.format(i1, target, i2), name)

            # if i1 == 'digit':
            #     dst = os.path.join('letters/{}/{}/{}'.format(i1,target, i2), name)
            # elif i1 == 'lower'
            #     dst = os.path.join('letters/{}/{}/{}'.format(i1,target, i2), name)
            # elif i1 == 'upper'
            #     dst = os.path.join('letters/{}/{}/{}'.format(i1,target, i2), name)
            # else:
            #     assert False

            src = os.path.join('my_letters', name)
            shutil.copyfile(src, dst)
        except:
            print(name)

    # print(collections.Counter(total))
    # Counter(
    #     {'digit0': 73, 'digit1': 73, 'digit2': 73, 'digit3': 73, 'digit4': 73, 'digit5': 73, 'digit6': 73, 'digit7': 73,
    #      'digit8': 73, 'digit9': 73, 'lowera': 71, 'lowerb': 71, 'lowerc': 71, 'uppera': 71, 'upperb': 71, 'upperc': 70,
    #      'lowerd': 61, 'lowere': 61, 'lowerf': 61, 'lowerg': 61, 'lowerh': 61, 'loweri': 61, 'lowerj': 61, 'upperh': 61,
    #      'upperi': 61, 'upperd': 60, 'uppere': 60, 'upperf': 60, 'upperg': 60, 'upperj': 60, 'lowerk': 58, 'lowerl': 58,
    #      'lowerm': 58, 'lowern': 58, 'lowero': 58, 'lowerp': 58, 'lowerq': 58, 'lowerr': 58, 'lowers': 58, 'lowert': 58,
    #      'loweru': 58, 'lowerw': 58, 'lowerx': 58, 'lowery': 58, 'upperk': 58, 'upperl': 58, 'upperm': 58, 'uppern': 58,
    #      'uppero': 58, 'upperp': 58, 'upperq': 58, 'upperr': 58, 'uppers': 58, 'uppert': 58, 'upperu': 58, 'upperw': 58,
    #      'upperx': 58, 'uppery': 58, 'upperz': 58, 'lowerv': 55, 'upperv': 55, 'lowerz': 48, 'digitc': 1, 'digitd': 1,
    #      'digite': 1, 'digitf': 1, 'digitg': 1, 'digitj': 1})


make_letter_folder()
copy_letters()