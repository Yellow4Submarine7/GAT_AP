import pandas as pd
import numpy as np
import pickle

def save_obj(obj, name ):
    with open('data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

input_file = "data/Musical_Instruments_change.csv"
data = pd.read_csv(input_file)

relation_list = []
member_dict = {}

#创建关系对
for row, column in data.iterrows():
    try:
        relation_tuple = (str(column['reviewerID']),str(column['asin']))
        relation_list.append(relation_tuple)
    except:
        print("第%d行出现错乱,错乱的评论内容为%s"%(column['num'],column['overall']))

#relation_list = [('AWCJ12KBO5VII', 'B00JBIVXGC'), ('A2Z7S8B5U4PAKJ', 'B00JBIVXGC'), ('A2WA8TDCTGUADI', 'B00JBIVXGC')]
#print(relation_list)

member_index = 0
for name_tuple in relation_list:
    for name in name_tuple:
        if name in member_dict:
            continue
        member_dict[name] = member_index
        member_index += 1

relation_matrix = [[0 for i in range(len(member_dict))]
                for i in range(len(member_dict))]

#save_obj(member_dict,'member_dict') #存储成员字典

for (x, y) in relation_list:
    x_index = member_dict[x]
    y_index = member_dict[y]
    relation_matrix[x_index][y_index] = 1 #此时是有向图，我们需要无向图
    relation_matrix[y_index][x_index] = 1
    

#print(relation_matrix)

#邻接矩阵转字典
adjacency_list_dict={i: np.nonzero(row)[0].tolist() for i, row in enumerate(relation_matrix)}
#print(A_dict)

for src_node, neighboring_nodes in adjacency_list_dict.items():
    for trg_node in neighboring_nodes:
        a = 1

#save_obj(adjacency_list_dict,'adjacency_list_dict')

print(member_dict['AWCJ12KBO5VII'])
k = 0
for i in relation_matrix:
    for j in i:
        if j == 1:
            k += 1
#print(k)    #10261 ojbk

#print(adjacency_list_dict[1])
