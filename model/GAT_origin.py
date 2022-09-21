import torch
import torch.nn as nn

"该模型除内部逻辑修改之外，还需要添加review_features 的dataloading代码"


class GAT(torch.nn.Module):
    '''
    这是使用的3个implementation的中的implementation3 
    '''
    def __init__(self, num_of_layers, num_heads_per_layer, 
    num_features_per_layer, add_skip_connection = True, bias = True,
    dropout = 0.6 , log_attention_weights = False):
        super().__init__()
        #在前面加一个第零层，head为1方便计算
        #主要是因为有第零层的初始features
        assert num_of_layers == len(num_heads_per_layer) == \
            len(num_features_per_layer) - 1, f'Enter valid arch params.'
        num_heads_per_layer = [1] + num_heads_per_layer
        gat_layers = []
        #设定各层参数
        for i in range(num_of_layers):
            #各层输入特征长度分别为[AMAZON_NUM_INPUT_FEATURES, 8, AMAZON_RATING_CLASSES][100,8,5]
            layer = GATLayer(
                num_in_features = num_features_per_layer[i] \
                    * num_heads_per_layer[i],
                num_out_features = num_features_per_layer[i + 1],
                num_of_heads = num_heads_per_layer[i + 1],
                #1到（n-1）层的多头concat，最后一层 mean avg
                concat = True if i < num_of_layers - 1 else False,
                #同上，最后一层分开处理
                activation = nn.ELU() if i < num_of_layers - 1 else None,
                dropout_prob = dropout,
                add_skip_connection = add_skip_connection,
                bias = bias,
                log_attention_weights = log_attention_weights
            )
            gat_layers.append(layer)
        #nn.Sequential是nn.module pytorch神经网络容器的一种构建方式
        #除了顺序方式之外还有nn.Modulelist和nn.Moduledict
        self.gat_net = nn.Sequential(
            *gat_layers,
        )
    
    def forward(self, data):
        return self.gat_net(data)

#具体定义每一个GAT层
class GATLayer(torch.nn.Module):
    src_nodes_dim = 0
    trg_nodes_dim = 1
    nodes_dim = 0
    head_dim = 1

    def __init__(self, num_in_features,num_out_features,num_of_heads,
                concat = True, activation = nn.ELU(), dropout_prob = 0.6, 
                add_skip_connection = True, bias = True, log_attention_weights \
                    = False):
        super().__init__()
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection
        '''
        设定可训练的权重矩阵有W，a，b
        
        全连接层定义待训练的中转矩阵W
        W矩阵使输入节点i的特征向量与输出节点j的特征向量
        能够concatenate
        Whi||Whj
        '''
        #注意这里的nn.Linear是一个类,linear_porj
        #是类的实例
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * 
                            num_out_features, bias = False)
        #作者使用hi及hj矩阵分别左乘和右乘a的方式再相加来计算原式中的e
        #eij = ei + ej = LeakyReLU(score_source + score_target)
        #而非论文中hi，hj concatenate再乘a的方式
        #nn.parameter设定可训练参数
        self.scoring_fn_review = nn.Parameter(torch.Tensor(1,num_of_heads,
                                num_out_features))
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1,num_of_heads, 
                                num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1,num_of_heads,
                                num_out_features))
        #（改）增加review边的可训练参数
        
        #设定bias 不是重点，论文中也没提及
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter("bias", None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, 
                            num_of_heads * num_out_features, bias = False)
        else:
            self.register_parameter("bias", None)
        #
        # trainable weights 设置结束
        #
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.activation = activation
        self.dropout = nn.Dropout(p = dropout_prob)
        self.log_attention_weights = log_attention_weights
        self.attention_weights = None
        self.init_params()

    def forward(self, data):
        #
        #Steop 1:划分多头
        #
        in_nodes_features, review_features, edge_index = data
        #num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        num_of_nodes = in_nodes_features.size()[self.nodes_dim]
        assert edge_index.shape[0] == 2, f"Expected \
            edge index with shape=(2,E) got {edge_index.shape}"
        'reviews_features的操作方式应该跟nodes_features的操作方式一样吗？是否也应该更新,每次生成out_reviews_features'
        '算了，暂时先不这样，全局共享同一个reviews_features好了'
        '不行，这样的话无法进行下层的计算'
        '以上这些思路都不对，nodes_features与review_features向量长度本来就不相同，应该concatenate,这样每次用原始的reviews_features也行'
        in_nodes_features = self.dropout(in_nodes_features)
        review_features = self.dropout(review_features)
        print("in_nodes_features的size是{}".format(in_nodes_features.size()))
        print("review_features的size是{}".format(review_features.size()))
        '''
        shape1 = (N,FIN)*(FIN, NH*FOUT)      (self.linear_proj)
        nn.linear 全连接层相当于做了一个XW+B = Y 
        这里输出Y的形状为(N,NH*FOUT)
        shape = (N,FIN)*(FIN,NH*FOUT)-> (N, NH, FOUT)
        where NH - number of heads, FOUT - num of output heatures
        N - number of nodes
        view 用于改变矩阵形状
        ''' 
        nodes_features_proj = self.linear_proj(in_nodes_features).\
            view(-1,self.num_of_heads,self.num_out_features)    
        #(改,reviews_features_proj需要自己的linear_proj函数,reviews_features永远不变，跟in和out长度无关)
        reviews_features_proj = self.linear_proj(review_features).\
            view(-1,self.num_of_heads,self.num_out_features)
        #??需要吗？droupout
        reviews_features_proj = self.dropout(reviews_features_proj)
        #print("nodes_features_proj的size是{}".format(nodes_features_proj.size()))
        #print("reviews_features_proj的size是{}".format(reviews_features_proj.size()))
        #
        #Step 2: 计算边的attention
        #
        '''
        Hadamard product 哈达玛积*不就是矩阵对应元素相乘嘛， element-wise product
        矩阵形状不变
        torch.sum对某一维度求和，对shape来说挤压这一维度变成1.?????为什么要这样
        '''
        scores_source = (nodes_features_proj*self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj*self.scoring_fn_target).sum(dim=-1)
        scores_review = (reviews_features_proj*self.scoring_fn_review).sum(dim=-1)
        #print("scores_target的size是{}".format(scores_target.size()))
        #scores_reivew = (edge_review_proj*self.scoring_fn_review).sum(dim=-1)
        #lift一下再通过源节点和目标节点分数求边的分数，lift应该是从所有的scorecs里面按照edge_index选出需要计算的边
        #后面给出函数定义，获取需要使用的那部分特征

        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = \
            self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        "#############需要写review_lift函数将此时需要的review_edge给lift出来#################"
        "no, 不对，不需要lift，lift是选出有边的节点，reviews本来就是边"
        "当前lift函数产生的连接包含了节点自身连接和AB、BA节点正反连接，产生了22851个连接。但只需要10261个连接"
        #scores_review = self.review_lift(scores_review, reviews_features_proj, edge_index)
        #########最重要的两个公式和步骤###########
        #eij = ei + ej = LeakyReLU(score_source+score_target)
        #(改)eij = ei + ej + edge(ij) = LeakyReLu(score_source+score_target+score_review)
        #score_source -> Whi , score_target -> Whj
        #(改)score_source -> Whi , score_target -> Whj, score_review -> Wc(ij)
        print("scores_source_lifted的size是{}".format(scores_source_lifted.size()))
        print("scores_target_lifted的size是{}".format(scores_target_lifted.size()))
        print("scores_review的size是{}".format(scores_review.size()))
        scores_per_edge = self.leakyReLU(scores_source_lifted+\
            scores_target_lifted+scores_review)
        #αij = softmax(eij)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, 
            edge_index[self.trg_nodes_dim], num_of_nodes)
        attentions_per_edge = self.dropout(attentions_per_edge)
        #最后一步，αij乘Whj 再求和得到 Whi
        #(改)最后一步，αij乘(Whj+Wc(ij)) 再求和得到新Whi
        #αij乘Whj
        #print("αij乘(Whj+Wc(ij))")
        #print("nodes_features_proj_lifted的size是{}".format(nodes_features_proj_lifted.size()))

        #print("attentions_per_edge的size是{}".format(attentions_per_edge.size()))
        nodes_features_proj_lifted_weighted_left = nodes_features_proj_lifted*attentions_per_edge
        nodes_features_proj_lifted_weighted_right = scores_review.unsqueeze(-1)*attentions_per_edge
        #nodes_features_proj_lifted_weighted = (nodes_features_proj_lifted*attentions_per_edge)+(scores_review*attentions_per_edge)
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted_weighted_right + nodes_features_proj_lifted_weighted_left
        #求和，注意这里的hj看起来是用初始的hj？没有用新的hj，如果用新的话，应该要递归
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted,
        edge_index,in_nodes_features,num_of_nodes)
        out_nodes_features = self.skip_concat_bias(attentions_per_edge, \
            in_nodes_features,out_nodes_features)
        return (out_nodes_features,review_features,edge_index)
    
    #Helper functions
    #αij = softmax(eij),就是用当前要计算的边ij比上所有的i的邻边
    '''
    节点3的邻边有1-3,2-3,3-3那么α1-3 = 1-3/(1-3 + 2-3 + 3-3)
    '''
    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        #Subtracting the max value from logits doesn't change the end 
        #result but it improves the numerical stability 
        #这里应该是减去多头里面最大的值
        #scores_per_edge应该是包括了图内所有边！！！(不是i的邻边)的矩阵，这个函数可同时计算出i的所有邻边attentnion
        #这些所有函数都是针对所有图上的节点一起计算的
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()
        
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge,
        trg_index, num_of_nodes)

        #1e-16 理论上不需要，加上是为了避免电脑把过小的数约等于0
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)
        
        #shape = (E,NH) -> (E,NH,1)
        return attentions_per_edge.unsqueeze(-1)
    
    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        #扩展维度方便计算，explicit_broadcast函数使用unsqueeze方法扩展维度
        #E -> (E,NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)
        #矩阵形状转换成列表元素(N,NH)N个节点,方便使用torch.zeros
        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype = exp_scores_per_edge.dtype, device = exp_scores_per_edge.device)
        #self_tensor.scatter_add_(dim,index_tensor,other_tensor)
        #把other_tensor中的元素加到self_tensor里面，加入位置有一维度(dim设定)按照index_tensor索引的位置设定
        #其他维与other_tensor中的位置一致 
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
        #index_select(dim,index)按照index的索引选出对应维度的元素，组成新矩阵
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[self.nodes_dim] = num_of_nodes
        out_nodes_features = torch.zeros(size, dtype = in_nodes_features.dtype, device = in_nodes_features.device)
        #用unsqueeze扩展维度（全添加1）        
        trg_index_broadcasted = self.explicit_broadcast(
            edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        out_nodes_features.scatter_add_(
            self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)
        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]
        '添加review_feature的索引'
        #pytorch index_select(dim, index)方法， 按照给定维度dim和索引index选择输入矩阵中的元素
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)
        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    #broadcast是一种在不同size的tensor进行加减自动运行的一种处理机制
    #具体机制为1.先将维度补全 2.自动调用Expand函数使各个维度size相同
    #Expand扩展时是采用复制某一维度元素到其他维度的办法
    #这里作者构建了一个显式的broadcast
    def explicit_broadcast(self, this, other):
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)
        return this.expand_as(other)

    #使用Glorot(Xavier)初始化方法
    #参数从[-limit, limit]的均匀分布产生，其中limit为sqrt(6 / (fan_in + fan_out))。fan_in为权值张量的输入单元数，fan_out是权重张量的输出单元数。
    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients,in_nodes_features, out_nodes_features):
        if self.log_attention_weights:
            self.attention_weights = attention_coefficients
        #？为什么加一下就能实现残差连接的功能？xxx
        #如果输入特征和输出特征的长度相同，就简单利用unsqueeze增加一维度后用broadcast（相加）把input vectors复制NH次
        #如果不同则是使用skip_proj实现残差连接，skip_proj是一个nn.Linear 全连接
        if self.add_skip_connection:
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:
                #(N,FIN) -> (N,1,FIN) -> (N, NH, FOUT)
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        #多头需要融合，还是平均
        if self.concat:
            #shape: (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            #shape: (N,NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim = self.head_dim)
        if self.bias is not None:
            out_nodes_features += self.bias
        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)






        

        


        


        
        
        
        




            


            

            
                       



            


            

            
            
            
            
        
            
            
            
            

            
            
            





    





        


        



        

    

            
    

