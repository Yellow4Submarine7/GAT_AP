#Musical_Instruments.csv的部分文件评分和评论位置错乱调整
import pandas as pd

input_file = "data/Musical_Instruments copy.csv"
output_file = "data/Musical_Instruments_change.csv"
data = pd.read_csv(input_file)

for index,row in data.iterrows():
    try:
        float(row['overall'])
    except:
        print("第%d行出现错乱,错乱的评论内容为%s"%(row['num'],row['overall'])) #既然是错乱的，那么就应该评分的位置是评论内容
        '''
        temp = row['overall']
        row['overall'] =  row['reviewText']
        row['reviewText'] = temp
################
        这样不对，只修改了当前row的值，row[column]是从data复制过来的，data本身没有被修改
        '''
        data['overall'][row['num']] = row['reviewText']
        data['reviewText'][row['num']] = row['overall']

#记得还要把dataframe存储回去csv
print("处理完成，写回csv")
data.to_csv(output_file,index=False,sep=',')
print("Done")


