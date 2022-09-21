from numpy import random
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import datetime

df_movies = pd.read_csv(os.path.join(os.getcwd(),"data/ml-25m/movies.csv"))
df_ratings = pd.read_csv(os.path.join(os.getcwd(),"data/ml-25m/ratings.csv"))

#pd.Series 按照index创建序列
#创建电影ID和名称的查找字典
movieId_to_name = pd.Series(df_movies.title.values, index = df_movies.movieId.values).to_dict()       
name_to_movieId = pd.Series(df_movies.movieId.values, index = df_movies.title).to_dict()

ratings_train, ratings_test = train_test_split(df_ratings,stratify = df_ratings['userId'],
                                                random_state = 15688, test_size = 0.30)

#print("Number of training data: " + str(len(ratings_train)))
#print("Number of test data: " + str(len(ratings_test)))

'''设定4分及以上为用户喜欢的项目，用户喜欢的所有电影分别看作一个词一起组成一句话，不喜欢的电影也组成一句话'''

def rating_splitter(ratings):
    #np.where 条件语句(condition,x,y) 满足输出x不满足输出y
    ratings['liked'] = np.where(ratings['rating']>=4, 1, 0)
    ratings['movieId'] = ratings['movieId'].astype('str')
    user_like = ratings.groupby(['liked', 'userId'])
    return ([user_like.get_group(group)['movieId'].tolist() for group in user_like.groups])

#防止出现警告
pd.options.mode.chained_assignment = None
splitted_movies = rating_splitter(ratings_train)

#开始训练
start = datetime.datetime.now()

model = Word2Vec(sentences = splitted_movies, epochs = 10, min_count = 5, workers = 4,
                    sg = 1, hs = 0, negative = 5, window = 9999999)

print("Time passed: " + str(datetime.datetime.now()-start))
model.save('model/iterm2vec.model')
del model





