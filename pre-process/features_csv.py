import pandas as pd
import numpy as np
import csv
import torch

from torch import random
#建立user和item的合并features.csv

input_file = "data/Musical_Instruments_change.csv"
output_file = "data/random_features.csv"

#输入原csv文件位置，获取邻接矩阵及字典
def get_adj_matrix(input_file):
    data = pd.read_csv(input_file)
    global relation_list,member_dict
    relation_list = []
    #member_dict记录某节点(项目或用户)的序号
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

    #获取member_dict字典
    member_index = 0
    for name_tuple in relation_list:
        for name in name_tuple:
            if name in member_dict:
                continue
            member_dict[name] = member_index
            member_index += 1

    #建立全0的matrix
    relation_matrix = [[0 for i in range(len(member_dict))]
                    for i in range(len(member_dict))]
                
    for (x, y) in relation_list:
        x_index = member_dict[x]
        y_index = member_dict[y]
        relation_matrix[x_index][y_index] = 1 #此时是有向图,我们需要无向图
        relation_matrix[y_index][x_index] = 1
    
    return relation_matrix

relation_matrix =get_adj_matrix(input_file)
#print(relation_matrix)

#邻接矩阵转字典
A_dict={i: np.nonzero(row)[0].tolist() for i, row in enumerate(relation_matrix)}
#print(A_dict)

def test(relation_matrix):
    k = 0
    for i in relation_matrix:
        for j in i:
            if j == 1:
                k += 1
            #print(k)    #10261 ojbk

print(member_dict['1384719342'])

id_list = []
num_list = []

random_features_list = []

a = torch.rand(10)
print(a)
#for id,num in member_dict.items():
    #id_list.append(str(id))
    #num_list.append(str(num))
    #random_features_list.append(torch.rand(10).numpy)

#df=pd.DataFrame({'num':num_list,'id':id_list,'features':random_features_list})
#df.to_csv(output_file,index = False)