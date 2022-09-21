#Pickle模块将对象转换为一种可以传输或存储的格式
import pickle
import zipfile
import json

import pandas as pd 

import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import scipy.sparse as sp
import torch
from torch.hub import download_url_to_file
from torch.utils.data import DataLoader, Dataset

from utils.constants import *
from utils.visualizations import plot_in_out_degree_distributions, visualize_graph

#用AMAZON替换CORA
def load_graph_data(training_config, device):
    dataset_name = 'AMAZON'
    layer_type = training_config['layer_type']
    should_visualize = training_config['should_visualize']
    if dataset_name == 'AMAZON':  # Cora citation network

        # shape = (N, FIN), where N is the number of nodes and FIN is the number of input features
        #Pickle模块将对象转换为一种可以传输或存储的格式
        "node_features.pkl里存储了torchsize为(2329,1)的tensor"
        node_features = torch.load('data/random_node_features.pkl')
        # shape = (N, 100) (2329,100)
        "用ratings替换labels"
        ratings_csv = pd.read_csv('data/Musical_Instruments_change.csv', usecols=['overall'])
        # shape = (N, number of neighboring nodes) <- this is a dictionary not a matrix!
        #adjacency_list_dict = pickle_read(os.path.join(CORA_PATH, 'adjacency_list.dict'))
        #adjacency_list_dict = pickle_read(os.path.join('data/member_dict.pkl'))
        # Normalize the features
        #node_features_csr = normalize_features_sparse(node_features_csv)
        num_of_nodes = node_features.size()[0]
        '加入review_features,是一个(M,features)的tensor'
        "shape = (M, 300) (10261,300)"
        #review_features_csv = pd.read_csv('data/sentence_vec.csv',usecols=['sentence_vec'])
        review_features = torch.load('data/random_reviews_features.pkl')
        adjacency_list_dict = get_adjacency_list_dict("data/Musical_Instruments_change.csv")

        if layer_type == LayerType.IMP3:
            # Build edge index explicitly (faster than nx ~100 times and as fast as PyGeometric imp but less complex)
            # shape = (2, E), where E is the number of edges, and 2 for source and target nodes. Basically edge index
            # contains tuples of the format S->T, e.g. 0->3 means that node with id 0 points to a node with id 3.
            topology = build_edge_index(adjacency_list_dict, num_of_nodes, add_self_edges=False)
        elif layer_type == LayerType.IMP2 or layer_type == LayerType.IMP1:
            # adjacency matrix shape = (N, N)
            topology = nx.adjacency_matrix(nx.from_dict_of_lists(adjacency_list_dict)).todense().astype(np.float)
            topology += np.identity(topology.shape[0])  # add self connections
            topology[topology > 0] = 1  # multiple edges not allowed
            topology[topology == 0] = -np.inf  # make it a mask instead of adjacency matrix (used to mask softmax)
            topology[topology == 1] = 0
        else:
            raise Exception(f'Layer type {layer_type} not yet supported.')

        # Note: topology is just a fancy way of naming the graph structure data
        # (be it in the edge index format or adjacency matrix)

        if should_visualize:  # network analysis and graph drawing
            plot_in_out_degree_distributions(topology, num_of_nodes, dataset_name)
            visualize_graph(topology, dataset_name)

        # Convert to dense PyTorch tensors

        # Needs to be long int type (in implementation 3) because later functions like PyTorch's index_select expect it
        "当前的edge_select即topology是计算了节点自身连接以及AB、BA正反连接，需要修改"
        "IMP3的节点是由build_edge_index()构造，自身连接有add_selfedge 开关"
        topology = torch.tensor(topology, dtype=torch.long if layer_type == LayerType.IMP3 else torch.float, device=device)
        "需要先转换成np.array再转换成tensor"
        ratings = np.array(ratings_csv)
        ratings = torch.tensor(ratings, dtype=torch.long, device=device)
        #node_labels = torch.tensor(node_labels_npy, dtype=torch.long, device=device)  # Cross entropy expects a long int
        #'node_features已经是tensors,不过放在list中，应该把它转换成一个二维的tensor'
        node_features = node_features
        'review_features已经是tensor'
        review_features = review_features
        #review_features = np.array(review_features_list)
        #node_features = np.expand_dims(review_features, axis=1)
        # Indices that help us extract nodes that belong to the train/val and test splits
        train_indices = torch.arange(AMAZON_TRAIN_RANGE[0], AMAZON_TRAIN_RANGE[1], dtype=torch.long, device=device)
        '这里把头尾索引转换成数列,而user和item本来就是list'
        val_indices = torch.arange(AMAZON_VAL_RANGE[0], AMAZON_VAL_RANGE[1], dtype=torch.long, device=device)
        test_indices = torch.arange(AMAZON_TEST_RANGE[0], AMAZON_TEST_RANGE[1], dtype=torch.long, device=device)
        #注意torch.arange和torch.Tensro函数的区别，torch.Tensor的dtype直接写在函数名上
        train_item_indices = torch.LongTensor(AMAZON_TRAIN_ITEM_LIST, device=device)
        train_user_indices = torch.LongTensor(AMAZON_TRAIN_USER_LIST, device=device)
        val_item_indices = torch.LongTensor(AMAZON_VAL_ITEM_LIST, device=device)
        val_user_indices = torch.LongTensor(AMAZON_VAL_USER_LIST, device=device)
        test_item_indices = torch.LongTensor(AMAZON_TEST_ITEM_LIST, device=device)
        test_user_indices = torch.LongTensor(AMAZON_TEST_USER_LIST, device=device)


        return node_features, review_features, ratings, topology, train_indices, val_indices, test_indices, \
            train_item_indices, train_user_indices, val_item_indices, val_user_indices, test_item_indices, test_user_indices

'似乎处理这样的GAT的数据不需要batch？也就不需要DataLoader？？'
'对不需要分批，GraphDataLoader(DataLoader),GraphDataset(Dataset)两个class是在PPI中被引用'
'不过可以学习一下Graph神经网络是怎么分批的'

def json_read(path):
    with open(path, 'r') as file:
        data = json.load(file)

    return data

# All Cora data is stored as pickle
def pickle_read(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data

def pickle_save(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def normalize_features_sparse(node_features_sparse):
    assert sp.issparse(node_features_sparse), f'Expected a sparse matrix, got {node_features_sparse}.'

    # Instead of dividing (like in normalize_features_dense()) we do multiplication with inverse sum of features.
    # Modern hardware (GPUs, TPUs, ASICs) is optimized for fast matrix multiplications! ^^ (* >> /)
    # shape = (N, FIN) -> (N, 1), where N number of nodes and FIN number of input features
    node_features_sum = np.array(node_features_sparse.sum(-1))  # sum features for every node feature vector

    # Make an inverse (remember * by 1/x is better (faster) then / by x)
    # shape = (N, 1) -> (N)
    node_features_inv_sum = np.power(node_features_sum, -1).squeeze()

    # Again certain sums will be 0 so 1/0 will give us inf so we replace those by 1 which is a neutral element for mul
    node_features_inv_sum[np.isinf(node_features_inv_sum)] = 1.

    # Create a diagonal matrix whose values on the diagonal come from node_features_inv_sum
    diagonal_inv_features_sum_matrix = sp.diags(node_features_inv_sum)

    # We return the normalized features.
    return diagonal_inv_features_sum_matrix.dot(node_features_sparse)

# Not used -> check out playground.py where it is used in profiling functions
def normalize_features_dense(node_features_dense):
    assert isinstance(node_features_dense, np.matrix), f'Expected np matrix got {type(node_features_dense)}.'

    # The goal is to make feature vectors normalized (sum equals 1), but since some feature vectors are all 0s
    # in those cases we'd have division by 0 so I set the min value (via np.clip) to 1.
    # Note: 1 is a neutral element for division i.e. it won't modify the feature vector
    return node_features_dense / np.clip(node_features_dense.sum(1), a_min=1, a_max=None)

def get_adjacency_list_dict(input_file):
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

    member_index = 0
    for name_tuple in relation_list:
        for name in name_tuple:
            if name in member_dict:
                continue
            member_dict[name] = member_index
            member_index += 1

    relation_matrix = [[0 for i in range(len(member_dict))]
                    for i in range(len(member_dict))]

    for (x, y) in relation_list:
        x_index = member_dict[x]
        y_index = member_dict[y]
        relation_matrix[x_index][y_index] = 1 #此时是有向图，我们需要无向图
        relation_matrix[y_index][x_index] = 1
    

    adjacency_list_dict={i: np.nonzero(row)[0].tolist() for i, row in enumerate(relation_matrix)}
    return adjacency_list_dict

#
def build_edge_index_origin(adjacency_list_dict, num_of_nodes, add_self_edges=True):
    source_nodes_ids, target_nodes_ids = [], []
    seen_edges = set()

    for src_node, neighboring_nodes in adjacency_list_dict.items():
        for trg_node in neighboring_nodes:
            # if this edge hasn't been seen so far we add it to the edge index (coalescing - removing duplicates)
            if (src_node, trg_node) not in seen_edges:  # it'd be easy to explicitly remove self-edges (Cora has none..)
                source_nodes_ids.append(src_node)
                target_nodes_ids.append(trg_node)

                seen_edges.add((src_node, trg_node))

    if add_self_edges:
        source_nodes_ids.extend(np.arange(num_of_nodes))
        target_nodes_ids.extend(np.arange(num_of_nodes))

    # shape = (2, E), where E is the number of edges in the graph
    edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))

    return edge_index
def build_edge_index(adjacency_list_dict, num_of_nodes, add_self_edges=True):
    source_nodes_ids, target_nodes_ids = [], []
    seen_edges = set()

    for src_node, neighboring_nodes in adjacency_list_dict.items():
        for trg_node in neighboring_nodes:
            # if this edge hasn't been seen so far we add it to the edge index (coalescing - removing duplicates)
            if (src_node, trg_node) not in seen_edges:  # it'd be easy to explicitly remove self-edges (Cora has none..)
                if (trg_node, src_node) not in seen_edges:      #AB连接与BA连接看成一条连接
                    source_nodes_ids.append(src_node)
                    target_nodes_ids.append(trg_node)

                seen_edges.add((src_node, trg_node))

    if add_self_edges:
        source_nodes_ids.extend(np.arange(num_of_nodes))
        target_nodes_ids.extend(np.arange(num_of_nodes))

    # shape = (2, E), where E is the number of edges in the graph
    edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))

    return edge_index

# Not used - this is yet another way to construct the edge index by leveraging the existing package (networkx)
# (it's just slower than my simple implementation build_edge_index())
def build_edge_index_nx(adjacency_list_dict):
    nx_graph = nx.from_dict_of_lists(adjacency_list_dict)
    adj = nx.adjacency_matrix(nx_graph)
    adj = adj.tocoo()  # convert to COO (COOrdinate sparse format)

    return np.row_stack((adj.row, adj.col))
