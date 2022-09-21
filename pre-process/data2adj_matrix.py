#将csv原数据转换成评分矩阵和邻接矩阵
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import numpy as np
#csr_matrix 专门处理user,item,rating这类数据，但是user，及item标号需要是数字,不能是字符串

#上面两个都是垃圾，直接用pandas的透视表pivot_table
input_file = "data/Musical_Instruments_change.csv"
df = pd.read_csv(input_file)
adj_matrix = df.pivot_table(values='num',index=['asin'],columns=['reviewerID'])
#rint(adj_matrix)

#A_dict={i: np.nonzero(row)[0].tolist() for i, row in enumerate(adj_matrix)}

def convert_to_bool(data):
    if(np.isnan(data)):
        #data = 0,函数里面不用赋值，用return
        return 0
    else:
        #data = 1
        return 1

#applymap不改变原来的矩阵，而是重新生成一个
#print(adj_matrix.applymap(convert_to_bool))

