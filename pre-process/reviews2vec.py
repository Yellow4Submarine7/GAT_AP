#BERT-pytorch
#reviews2vec and append to csv
from matplotlib.pyplot import get
from transformers import BertModel, BertTokenizer, BertConfig
import torch
import numpy as np
#防止出现训练警告(这是预测，所以不需要)
from transformers import logging
logging.set_verbosity_error()
import os
import csv
import pandas as pd
from tqdm import tqdm


tokenizer = BertTokenizer.from_pretrained('/Users/brownjack/Downloads/bert-base-uncased/model')
model = BertModel.from_pretrained('/Users/brownjack/Downloads/bert-base-uncased/model',)

def get_sentence_vec(sentence):
    text_dict = tokenizer.encode_plus(sentence,max_length=512, add_special_tokens=True, return_attention_mask=True)
#升维
    input_ids = torch.tensor(text_dict['input_ids']).unsqueeze(0)
    token_type_ids = torch.tensor(text_dict['token_type_ids']).unsqueeze(0)
    attention_mask = torch.tensor(text_dict['attention_mask']).unsqueeze(0)
    res = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    #return res[0].detach().squeeze(0).numpy()
    return res[0].squeeze(0).detach()

#测试代码
sentences = "I love you"
features = get_sentence_vec(sentences).squeeze(0)
features = features.squeeze(0)
print(features.shape)

inputfile = "data/Musical_Instruments copy.csv"
outputfile = 'data/sentence_vec.pkl'
df = pd.read_csv(inputfile)

data = []
for index,row in tqdm(df.iterrows()):#使用tqdm追踪dataframe的迭代情况
    try:#防止reviewText为空
        data.append(get_sentence_vec(row['reviewText']))
    except:
        data.append(get_sentence_vec(""))
        print('{} row reivews_text = NaN'.format(index))


torch.save(data, 'sentence_vec.pkl')