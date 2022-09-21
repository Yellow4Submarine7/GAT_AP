import pandas as pd
import torch

node_features_list = torch.load('data/new_node_features.pkl')
for i in node_features_list:
    print(i)