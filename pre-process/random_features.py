import pandas as pd
import torch
#用torch自带的save和load读取和存储tensor！！！！！
#GAT.py需要的是(N,Features)形状的一个tensor
#首先要把featres tensor变成(N,1)形状，再把装有这些tensor的tensor list转换成整个tensor
input_file = "data/Musical_Instruments_change.csv"
output_file = "data/random_reviews_features.pkl"
data = pd.read_csv(input_file)

random_features_list = []

#其实就是获取一个遍历的长度
for index,row in data.iterrows():
    feature = torch.randn(10,10)
    feature = feature.view(-1,1)
    random_features_list.append(feature)

random_features_tensor = torch.stack(random_features_list, 0)
random_features_tensor = random_features_tensor.squeeze(dim=-1)
print(random_features_tensor)

print("处理完成,写入文件")

#torch.save(random_features_tensor, output_file)
print('写入完成')