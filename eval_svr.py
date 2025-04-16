import argparse
import torch
import time
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm
from data import MyDataset
from torch.utils.data import DataLoader
import pickle
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.svm import SVR
import logging

logging.basicConfig(
    format='%(message)s',
    level=logging.DEBUG,
)

# iForest
def iforest_partitioning(X, t, c):
    X = np.concatenate(X, axis=0)
    models = []
    for _ in range(t):
        sample_index = np.random.permutation(len(X))
        sample_index = sample_index[0: c]
        Y = [i for i in range(c)]
        sample_X = X[sample_index, :]
        model = DecisionTreeClassifier(splitter='random')
        model.fit(sample_X, Y)
        models.append(model)
    return models

# K-Means
def kmeans_partitioning(X, t, c, random_state=42):
    X = np.concatenate(X, axis=0)
    models = []
    for i in range(t):
        model = KMeans(n_clusters=c, init='random', random_state=random_state + i).fit(X)
        models.append(model)
    return models


def get_partition_index(partitions, X):
    partition_indexs = []
    t = len(partitions)
    num_graph = len(X)
    X_concate = np.concatenate(X, axis=0)

    graph_partition_indexs = []
    for i in range(t):
        model = partitions[i]
        partition_index = model.predict(X_concate)
        graph_partition_indexs.append(partition_index)
    graph_partition_indexs = np.array(graph_partition_indexs)

    flag = 0
    for i in range(num_graph):
        point_num = X[i].shape[0]
        partition_indexs.append(graph_partition_indexs[:, flag:flag + point_num])
        flag += point_num
    return partition_indexs


def get_pk_vectors(partition_indexs, c):
    pk_vectors = []
    [t, _] = partition_indexs[0].shape
    for partition_index in partition_indexs:
        num_point = partition_index.shape[1]
        graph_vectors = np.zeros((num_point, c * t))
        for i in range(num_point):
            instance_vector = np.zeros((1, c * t))
            instance_index = partition_index[:, i].flatten()
            bias = [t * c for t in range(t)]
            instance_index = instance_index + bias
            instance_vector[0, instance_index.flatten().tolist()] += 1
            graph_vectors[i, :] = instance_vector
        pk_vectors.append(graph_vectors)
    return pk_vectors


def custom_grid_search_cv_reg(model, param_grid, precomputed_kernels, y, cv=5):
    """
    parameters:
      model: SVR，kernel is set to 'precomputed'
      param_grid:  [{'C': ..., 'epsilon': ...}]
      precomputed_kernels: the precomputed kernel
      y: continuous label
      cv: the fold number for cross validation
    return:
      the best model and parameters 
    """
    cv_inner = KFold(n_splits=cv, shuffle=False)
    results = []
    params_list = []
    
    for train_idx, test_idx in cv_inner.split(precomputed_kernels[0]):
        fold_scores = []
        for K_idx, K in enumerate(precomputed_kernels):
            for p in list(ParameterGrid(param_grid)):
                # MAE
                sc = _fit_and_score(clone(model), K, y,
                                    scorer=make_scorer(mean_absolute_error),
                                    train=train_idx, test=test_idx,
                                    verbose=0, parameters=p, fit_params=None)
                fold_scores.append(sc.get('test_scores'))
                params_list.append({'K_idx': K_idx, 'params': p})
        results.append(fold_scores)
    results = np.array(results)

    mean_scores = results.mean(axis=0)
    best_idx = np.argmin(mean_scores)
    best_kernel_idx = params_list[best_idx]['K_idx']
    best_params = params_list[best_idx]['params']
    best_model = clone(model).set_params(**best_params)
    best_model.fit(precomputed_kernels[best_kernel_idx], y)
    return best_model, params_list[best_idx]


def run_regression(kernel_matrices, y, gridsearch=True, crossvalidation=True, 
                   param_grid=None, random_state=42):
    """
    parameters:
      kernel_matrices: to be computed
      y: continuous labels (array)
      gridsearch: weather or not to use grid search
      crossvalidation: whether to use cross validation or not
      param_grid: the search range for parameters C and epsilon
    return:
      The MAE value in each fold the best parameters
    """
    if param_grid is None:
        param_grid = [{'C': np.logspace(-3, 3, num=7), 'epsilon': [0.1]}]
    
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=random_state)
    performance_scores = []
    best_C = []
    best_epsilon = []
    
    with tqdm(total=cv_outer.get_n_splits(), desc="Cross-validation SVR Progress") as pbar:
        for train_idx, test_idx in cv_outer.split(kernel_matrices[0]):
            K_train = [K[train_idx][:, train_idx] for K in kernel_matrices]
            K_test = [K[test_idx][:, train_idx] for K in kernel_matrices]
            y_train, y_test = y[train_idx], y[test_idx]
            
            if gridsearch:
                svr = SVR(kernel='precomputed')
                gs, best_info = custom_grid_search_cv_reg(svr, param_grid, K_train, y_train, cv=5)
                C_ = best_info['params']['C']
                epsilon_ = best_info['params']['epsilon']
                y_pred = gs.predict(K_test[best_info['K_idx']])
            else:
                svr = SVR(C=1000.0, epsilon=0.1, kernel='precomputed')
                gs = svr.fit(K_train[0], y_train)
                y_pred = gs.predict(K_test[0])
                C_, epsilon_ = 1.0, 0.1
            
            best_C.append(C_)
            best_epsilon.append(epsilon_)
            
            score = mean_absolute_error(y_test, y_pred)
            performance_scores.append(score)
            if not crossvalidation:
                break
            
            pbar.update(1)

    if crossvalidation:
        print('Mean 10-fold MAE: {:.4f} ± {:.4f}'.format(np.mean(performance_scores), np.std(performance_scores)))
    else:
        print('Final MAE: {:.4f}'.format(np.mean(performance_scores)))
        
    return performance_scores, best_C, best_epsilon


# def svr_crossvalidation(kernel_matrix, train, val, test, y, C=np.logspace(-2, 3, 6)):
#     import warnings
#     from sklearn.exceptions import ConvergenceWarning
#     warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

#     if C is None:
#         C = np.logspace(-2, 3, 6)
#     param_grid = [{
#             'C': C
#         }]
    
#     K_train = kernel_matrix[:train, :train]
#     K_val = kernel_matrix[train:train+val, :train]
#     K_test = kernel_matrix[train+val:, :train+val]
#     y_train = y[:train]
#     y_val = y[train:train+val]
#     y_test = y[train+val:]
    
    
#     best_C = []
#     best_val_mae = float('inf')
#     max_iter = 1000000
#     for p in list(ParameterGrid(param_grid)):
#         model = SVR(kernel='precomputed', C=p['C'], cache_size=50, max_iter=max_iter)
#         model.fit(K_train, y_train)
#         val_mae = np.mean(np.abs(model.predict(K_val) - y_val))
#         # print(f'param: {p}, val MAE: {val_mae}')
#         if val_mae < best_val_mae:
#             best_val_mae = val_mae
#             best_C = p['C']
#     model = SVR(kernel='precomputed', C=best_C)
    
#     model.fit(kernel_matrix[:train+val, :train+val], y[:train+val])
#     y_pred = model.predict(K_test)
#     test_mae = np.abs(y_pred - y_test)
#     print(f'Best C: {best_C}, test MAE: {np.mean(test_mae):.3f} +- {np.std(test_mae):.3f}')
#     return best_C, np.mean(test_mae), np.std(test_mae)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ZINC',
                        help='name of dataset (default: ZINC)')
    parser.add_argument('--depth', type=int, default=4,
                        help='depth of bfs trees')
    parser.add_argument('--t', type=int, default=50,
                        help='number of partitionings')
    parser.add_argument('--c', type=int, default=4,
                        help='dim for pk')
    parser.add_argument('--partitioning', type=str, default='iforest', choices=['iforest', 'kmeans'],
                        help='partitioning method')
    parser.add_argument('--tokenizer_file', type=str, default='./tokenizer/ZINC/depth_7_tokenizer.json',
                        help='pretrained tokenizer file')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--model_file', type=str, default='./model/ZINC/best_model.pkl',
                        help='pretrained model file')
    parser.add_argument('--load_embedding', default=False, action='store_true',
                        help='use saved shortest-path embeddings')
    parser.add_argument('--crossvalidation', default=True, action='store_false',
                        help='enable 10 fold cross validation')
    parser.add_argument('--gridsearch', default=False, action='store_true',
                        help='enable grid search')

    
    args = parser.parse_args()
    
    # logging.info(f'dataset={args.dataset}, k={args.depth}')
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    if args.dataset == 'ZINC':
        # logging.info('Load data from .pkl file.')
        data_path = './dataset_ori/ZINC/ZINC.pkl'
        
        # import data from .pkl file
        with open(data_path, 'rb') as f:
            train_graphs, val_graphs, test_graphs = pickle.load(f)
        graphs = train_graphs + val_graphs + test_graphs
        graph_num = len(graphs)
    
        train, val, test = len(train_graphs), len(val_graphs), len(test_graphs)
        y = [graph.graph['label'] for graph in graphs]
    
    if not args.load_embedding:
        tokenizer = Tokenizer.from_file(args.tokenizer_file)
        model = torch.load(args.model_file, map_location='cpu')
        model.eval()
    
    # test
    # graphs = graphs[:100]
    
    time_start = time.time()
    
    if args.load_embedding:
        ms_graph_embeddings = pickle.load(open(f'./sp_embs/{args.dataset}/emb_{args.depth}.pkl', 'rb'))
    else:
        node_embeddings = []
        for k in range(args.depth):
            corpus_files = [f'./corpus/{args.dataset}/sp_depth_{k+1}']
            
            all_data = MyDataset(corpus_files, tokenizer)    # max_len = k + 1
            
            if args.batch_size == -1:
                my_loader = DataLoader(dataset=all_data, batch_size=len(all_data))
            else:
                my_loader = DataLoader(dataset=all_data, batch_size=args.batch_size)
            
            outs = []
            for src in my_loader:
                tgt = src[:, :-1]
                out = model(src, tgt)
                # remove <bos> <eos> <pad>
                tgt_key_padding_mask = torch.ones(tgt.size())
                tgt_key_padding_mask[tgt == 1] = 0
                tgt_key_padding_mask[tgt == 2] = 0
                tgt_key_padding_mask[tgt == 3] = 0
                tgt_key_padding_mask = tgt_key_padding_mask.unsqueeze(2).broadcast_to(out.size())
                out = out * tgt_key_padding_mask
                outs.extend(torch.sum(out, dim=1).tolist())
            node_embedding = np.array(outs)
            node_embeddings.append(node_embedding)
        ms_node_embeddings = scale(np.array(np.concatenate(node_embeddings, axis=1)), axis=0)
        # logging.info(f'Shape of node features: {ms_node_embeddings.shape}')
    
        ms_graph_embeddings = []
        index = 0
        num_features = len(ms_node_embeddings[0])
        for gidx in range(graph_num):
            n = len(graphs[gidx])
            feature_matrix = np.zeros((n, num_features))
            for node in range(n):
                feature_matrix[node,:] = ms_node_embeddings[node+index,:]
            index += n
            ms_graph_embeddings.append(feature_matrix)
        
    
    partitioning = 'iforest' if args.partitioning == 'iforest' else 'kmeans'
    
    c = args.c if partitioning == 'iforest' else int(args.c * graph_num)
    
    # logging.info(f't={args.t}, c={c}')
    np.random.seed(42)
    
    if partitioning == 'iforest':
        partitions = iforest_partitioning(ms_graph_embeddings, args.t, c)
    else:
        partitions = kmeans_partitioning(ms_graph_embeddings, args.t, c)
    partition_indexs = get_partition_index(partitions, ms_graph_embeddings)
    pk_vectors = get_pk_vectors(partition_indexs, c)
    graph_features = np.array([np.mean(c, axis=0) for c in pk_vectors])
    
    # compute kernel matrix
    kernel_matrix = graph_features @ graph_features.T
    diagonal_elements = np.diag(kernel_matrix)
    normalization_factors = np.sqrt(np.outer(diagonal_elements, diagonal_elements))
    kernel_matrix = kernel_matrix / normalization_factors
    
    time_end = time.time()
    
    logging.info(f'Kernel matrix is computed in {round(time_end - time_start, 2)}s.')
    
    # svr
    param_grid = {
        'C': np.logspace(-3, 3, 7),
        'epsilon': [0.1]
    }
    run_regression([kernel_matrix], np.array(y), args.gridsearch, args.crossvalidation, param_grid)
    
