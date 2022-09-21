from numpy import random
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import datetime

df = pd.read_csv(os.path.join(os.getcwd(),"data/Musical_Instruments_change.csv"))
data_train, data_test = train_test_split(df,stratify = df['reviewerID'],
                                                random_state = 15688, test_size = 0.30)
print("Number of training data: " + str(len(data_train)))
print("Number of test data: " + str(len(data_test)))

'''设定4分及以上为用户喜欢的项目，用户喜欢的所有项目分别看作一个词一起组成一句话，不喜欢的电影也组成一句话'''
def rating_splitter(data):
    #np.where 条件语句(condition,x,y) 满足输出x不满足输出y
    data['liked'] = np.where(data['overall'].astype('float64')>=4, 1, 0)
    data['asin'] = data['asin'].astype('str')
    reviwer_like = data.groupby(['liked', 'asin'])
    return ([reviwer_like.get_group(group)['asin'].tolist() for group in reviwer_like.groups])
#防止出现警告
pd.options.mode.chained_assignment = None
splitted_data = rating_splitter(data_train)
print(splitted_data)
#开始训练
#start = datetime.datetime.now()

#model = Word2Vec(sentences = splitted_data, epochs = 10, min_count = 5, workers = 4,
                    #sg = 1, hs = 0, negative = 5, window = 9999999)

#print("Time passed: " + str(datetime.datetime.now()-start))
#model.save('model/amazon_iterm2vec.model')
#del model





