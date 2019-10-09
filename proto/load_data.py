import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from logging import getLogger

import os

TRAIN_DATA = '../input/train.csv'
TEST_DATA = '../input/test.csv'

logger = getLogger(__name__)

def read_csv(path):
    logger.debug('enter')
    df = pd.read_csv(path,encoding='cp932')#encoding='cp932'はutf8を開くときのおまじない
    logger.debug('exit')
    return df

def load_train_data():
    logger.debug('enter')
    df = read_csv(TRAIN_DATA)
    df = df.rename(columns={'賃料':'target', '契約期間':'Contract', '間取り':'Room', 
                              '築年数':'Passed', '駐車場':'Parking', '室内設備':'Facility', 
                              '放送・通信':'Internet', '周辺環境':'Building', '建物構造':'Material', 
                              '面積':'Area', 'キッチン':'Kitchen', '所在地':'Place',
                              'バス・トイレ':'Bath', '所在階':'Floor', 'アクセス':'Access', 
                              '方角':'Angle'})
    logger.debug('exit')
    return df

def load_test_data():
    logger.debug('enter')
    df = read_csv(TEST_DATA)
    df = df.rename(columns={'契約期間':'Contract', '間取り':'Room', '築年数':'Passed', 
                            '駐車場':'Parking', '室内設備':'Facility', '放送・通信':'Internet', 
                            '周辺環境':'Building', '建物構造':'Material', '面積':'Area', 
                            'キッチン':'Kitchen', '所在地':'Place', 'バス・トイレ':'Bath', 
                            '所在階':'Floor', 'アクセス':'Access', '方角':'Angle'})
    logger.debug('exit')
    return df

if __name__ == '__main__':
    
    df_train = load_train_data()

    x_train = df_train.drop('target', axis=1)
    y_train = df_train['target']
    
    df_train.describe()
    sns.distplot(y_train),plt.show()

    y_train.sort_values(ascending=False).head(10)
