from os import kill
import pandas as pd
import pickle
import numpy as np

def save_obj(obj, name ):
    with open('data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

rating_file = "data/Musical_Instruments_change.csv"
ratings_data = pd.read_csv(rating_file)

item_name_list = []
user_name_list = []
for index,row in ratings_data.iterrows():
    user_name_list.append(row['reviewerID'])
    item_name_list.append(row['asin'])

#查成员字典

member_dict = load_obj('member_dict')
item_indices = []
user_indices = []

k = 0
for i in item_name_list:
    item_indices.append(member_dict[i])
    k += 1
print(k)

l = 0
for j in user_name_list:
    user_indices.append(member_dict[j])
    l += 1
print(l)

#转换成numpyarray再使用numpy的方法存储
item_indices = np.array(item_indices)
user_indices = np.array(user_indices)
filename = 'data/item_indices.npy'
np.save(filename, item_indices)

filename = 'data/user_indices.npy'
np.save(filename, user_indices)