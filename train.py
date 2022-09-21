import argparse
from inspect import classify_class_attrs
import time
from networkx.classes.function import nodes


import torch
import torch.nn as nn
from torch.optim import Adam


from model.GAT import GAT
from utils.data_loading import load_graph_data
from utils.constants import *
import utils.util as util

# Simple decorator function so that I don't have to pass arguments that don't change from epoch to epoch
def get_main_loop(config, gat, cross_entropy_loss, optimizer, node_features, review_features,ratings, edge_index, train_indices, val_indices,test_indices\
                ,train_item_indices, train_user_indices, val_item_indices, val_user_indices, test_item_indices, test_user_indices\
                ,patience_period, time_start):

    node_dim = 0  # node axis

    #train_labels = node_labels.index_select(node_dim, train_indices)
    #val_labels = node_labels.index_select(node_dim, val_indices)
    #test_labels = node_labels.index_select(node_dim, test_indices)

    train_ratings = ratings.index_select(node_dim, train_indices)
    val_ratings = ratings.index_select(node_dim, val_indices)
    test_ratings = ratings.index_select(node_dim, test_indices)

    # node_features shape = (N, FIN), edge_index shape = (2, E)
    '加入review_features'
    graph_data = (node_features,review_features,edge_index)  # I pack data into tuples because GAT uses nn.Sequential which requires it
    print(node_features)

    # def get_node_indices(phase):
    #     if phase == LoopPhase.TRAIN:
    #         return train_indices
    #     elif phase == LoopPhase.VAL:
    #         return val_indices
    #     else:
    #         return test_indices
    
    def get_item_indices(phase):
        if phase == LoopPhase.TRAIN:
            return train_item_indices
        elif phase == LoopPhase.VAL:
            return val_item_indices
        else:
            return test_item_indices       

    def get_user_indices(phase):
        if phase == LoopPhase.TRAIN:
            return train_user_indices
        elif phase == LoopPhase.VAL:
            return val_user_indices
        else:
            return test_user_indices   

    def get_ratings(phase):
        if phase == LoopPhase.TRAIN:
            return train_ratings
        elif phase == LoopPhase.VAL:
            return val_ratings
        else:
            return test_ratings

    def main_loop(phase, epoch=0):
        global BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT, writer

        # Certain modules behave differently depending on whether we're training the model or not.
        # e.g. nn.Dropout - we only want to drop model weights during the training.
        "这里只是将模型设置为train或eval模式，并没有真正训练"
        if phase == LoopPhase.TRAIN:
            gat.train()
        else:
            gat.eval()

        #node_indices = get_node_indices(phase)
        "node_indicec应该是没用了,因为索引用于提取节点来参与trian val test，这三个部分现在仅与user和item及rating有关"
        item_indices = get_item_indices(phase)
        #print(item_indices.size())
        user_indices = get_user_indices(phase)
        #print(user_indices.size())
        gt_ratings = get_ratings(phase).squeeze(-1)  # gt stands for ground truth
        #print(gt_ratings.size())

        # Do a forwards pass and extract only the relevant node scores (train/val or test ones)
        # Note: [0] just extracts the node_features part of the data (index 1 contains the edge_index)
        # shape = (N, C) where N is the number of nodes in the split (train/val/test) and C is the number of classes
        "这句开始正式进行训练,同时也获取了计算loss所需的预测scores"
        "gat获取一个graph_data并输出一个graph_data"
        "这样的graph_data只需要包含所有node features和拓扑结构"
        "应该改写index_select来一次性获取item_scores和user_scres并计算所需的预测score"
        "这样的话应该先获得new_graph_data，再对该数据进行操作获取item_scores和user_scores,最后计算获得预测score"
        "关键是item_indices和user_indices,应该从10261条交易记录里选, 但给的是2329个node features"
        "所以item_indices就是根据交易记录从features表里查序号,user_indices同理"
        #nodes_unnormalized_scores = gat(graph_data)[0].index_select(node_dim, node_indices)
        new_node_features = gat(graph_data)[0]
        torch.save(new_node_features, "data/new_node_features.pkl")
        item_features = new_node_features.index_select(node_dim, item_indices)
        user_features = new_node_features.index_select(node_dim, user_indices)
        #print(item_features)
        #predict_unnormalized_scores = item_features*user_features
        predict_unnormalized_scores = user_features*item_features
        #print(predict_unnormalized_scores.size())

        # Example: let's take an output for a single node on Cora - it's a vector of size 7 and it contains unnormalized
        # scores like: V = [-1.393,  3.0765, -2.4445,  9.6219,  2.1658, -5.5243, -4.6247]
        # What PyTorch's cross entropy loss does is for every such vector it first applies a softmax, and so we'll
        # have the V transformed into: [1.6421e-05, 1.4338e-03, 5.7378e-06, 0.99797, 5.7673e-04, 2.6376e-07, 6.4848e-07]
        # secondly, whatever the correct class is (say it's 3), it will then take the element at position 3,
        # 0.99797 in this case, and the loss will be -log(0.99797). It does this for every node and applies a mean.
        # You can see that as the probability of the correct class for most nodes approaches 1 we get to 0 loss! <3
        #loss = cross_entropy_loss(nodes_unnormalized_scores, gt_node_labels)
        #print(gt_ratings)
        #print(predict_unnormalized_scores)
        loss = cross_entropy_loss(predict_unnormalized_scores,gt_ratings)

        if phase == LoopPhase.TRAIN:
            optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph (.grad fields)
            loss.backward()  # compute the gradients for every trainable weight in the computational graph
            optimizer.step()  # apply the gradients to weights

        # Calculate the main metric - accuracy

        # Finds the index of maximum (unnormalized) score for every node and that's the class prediction for that node.
        # Compare those to true (ground truth) labels and find the fraction of correct predictions -> accuracy metric.
        #class_predictions = torch.argmax(nodes_unnormalized_scores, dim=-1)
        class_predictions = torch.argmax(predict_unnormalized_scores, dim=-1)
        accuracy = torch.sum(torch.eq(class_predictions, gt_ratings).long()).item() / len(gt_ratings)

        #
        # Logging
        #

        if phase == LoopPhase.TRAIN:
            # Log metrics
            if config['enable_tensorboard']:
                writer.add_scalar('training_loss', loss.item(), epoch)
                writer.add_scalar('training_acc', accuracy, epoch)

            # Save model checkpoint
            if config['checkpoint_freq'] is not None and (epoch + 1) % config['checkpoint_freq'] == 0:
                ckpt_model_name = f'gat_{config["dataset_name"]}_ckpt_epoch_{epoch + 1}.pth'
                config['test_perf'] = -1
                torch.save(util.get_training_state(config, gat), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))

        elif phase == LoopPhase.VAL:
            # Log metrics
            if config['enable_tensorboard']:
                writer.add_scalar('val_loss', loss.item(), epoch)
                writer.add_scalar('val_acc', accuracy, epoch)

            # Log to console
            if config['console_log_freq'] is not None and epoch % config['console_log_freq'] == 0:
                print(f'GAT training: time elapsed= {(time.time() - time_start):.2f} [s] | epoch={epoch + 1} | val acc={accuracy}')

            # The "patience" logic - should we break out from the training loop? If either validation acc keeps going up
            # or the val loss keeps going down we won't stop
            if accuracy > BEST_VAL_PERF or loss.item() < BEST_VAL_LOSS:
                BEST_VAL_PERF = max(accuracy, BEST_VAL_PERF)  # keep track of the best validation accuracy so far
                BEST_VAL_LOSS = min(loss.item(), BEST_VAL_LOSS)  # and the minimal loss
                PATIENCE_CNT = 0  # reset the counter every time we encounter new best accuracy
            else:
                PATIENCE_CNT += 1  # otherwise keep counting

            if PATIENCE_CNT >= patience_period:
                raise Exception('Stopping the training, the universe has no more patience for this training.')

        else:
            return accuracy  # in the case of test phase we just report back the test accuracy

    return main_loop  # return the decorated function


def train_gat_amazon(config):
    global BEST_VAL_PERF, BEST_VAL_LOSS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    # Step 1: load the graph data
    node_features, review_features, ratings, edge_index, train_indices, val_indices, test_indices\
        , train_item_indices, train_user_indices, val_item_indices, val_user_indices\
        , test_item_indices, test_user_indices = load_graph_data(config, device)

    # Step 2: prepare the model
    gat = GAT(
        num_of_layers=config['num_of_layers'],
        num_heads_per_layer=config['num_heads_per_layer'],
        num_features_per_layer=config['num_features_per_layer'],
        add_skip_connection=config['add_skip_connection'],
        bias=config['bias'],
        dropout=config['dropout'],
        #layer_type=config['layer_type'],
        log_attention_weights=False  # no need to store attentions, used only in playground.py for visualizations
    ).to(device)

    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(gat.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    main_loop = get_main_loop(
        config,
        gat,
        loss_fn,
        optimizer,
        node_features,
        review_features,
        ratings,
        edge_index,
        train_indices,
        val_indices,
        test_indices,
        train_item_indices, 
        train_user_indices, 
        val_item_indices, 
        val_user_indices, 
        test_item_indices, 
        test_user_indices,
        config['patience_period'],
        time.time())

    BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT = [0, 0, 0]  # reset vars used for early stopping

    # Step 4: Start the training procedure
    for epoch in range(config['num_of_epochs']):
        # Training loop
        main_loop(phase=LoopPhase.TRAIN, epoch=epoch)

        # Validation loop
        with torch.no_grad():
            try:
                main_loop(phase=LoopPhase.VAL, epoch=epoch)
            except Exception as e:  # "patience has run out" exception :O
                print(str(e))
                break  # break out from the training loop

    # Step 5: Potentially test your model
    # Don't overfit to the test dataset - only when you've fine-tuned your model on the validation dataset should you
    # report your final loss and accuracy on the test dataset. Friends don't let friends overfit to the test data. <3
    if config['should_test']:
        test_acc = main_loop(phase=LoopPhase.TEST)
        config['test_perf'] = test_acc
        print(f'Test accuracy = {test_acc}')
    else:
        config['test_perf'] = -1

    # Save the latest GAT in the binaries directory
    torch.save(
        util.get_training_state(config, gat),
        os.path.join(BINARIES_PATH, util.get_available_binary_name(config['dataset_name']))
    )


def get_training_args():
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=10000)
    parser.add_argument("--patience_period", type=int, help="number of epochs with no improvement on val before terminating", default=1000)
    parser.add_argument("--lr", type=float, help="model learning rate", default=0.00000001)   #默认为5e-3
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=5e-4)
    parser.add_argument("--should_test", action='store_true', help='should test the model on the test dataset? (no by default)')

    # Dataset related
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='dataset to use for training', default=DatasetType.CORA.name)
    parser.add_argument("--should_visualize", action='store_true', help='should visualize the dataset? (no by default)')

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", action='store_true', help="enable tensorboard logging (no by default)", default='yes')
    parser.add_argument("--console_log_freq", type=int, help="log to output console (epoch) freq (None for no logging)", default=100)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq (None for no logging)", default=1000)
    args = parser.parse_args()

    # Model architecture related
    gat_config = {
        "num_of_layers": 2,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_heads_per_layer": [8, 1],
        "num_features_per_layer": [AMAZON_NUM_INPUT_FEATURES, 8, AMAZON_RATING_CLASSES],
        "add_skip_connection": False,  # hurts perf on Cora
        "bias": True,  # result is not so sensitive to bias
        "dropout": 0.6,  # result is sensitive to dropout
        "layer_type": LayerType.IMP3  # fastest implementation enabled by default
    }

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    # Add additional config information
    training_config.update(gat_config)

    return training_config


if __name__ == '__main__':

    # Train the graph attention network (GAT)
    train_gat_amazon(get_training_args())
