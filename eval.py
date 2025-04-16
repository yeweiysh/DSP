import argparse
import torch
import time
import scipy.io as sio
import numpy as np
from tokenizers import Tokenizer
from data import MyDataset
from torch.utils.data import DataLoader
import pickle
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.svm import SVC
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


def grid_search(model, param_grid, precomputed_kernels, y, cv=5, random_state=None):
    cv = StratifiedKFold(n_splits=cv, shuffle=True)
    results = []
    for train_index, test_index in cv.split(precomputed_kernels[0], y):
        split_results = []
        params = []
        for K_idx, K in enumerate(precomputed_kernels):
            for p in list(ParameterGrid(param_grid)):
                sc = _fit_and_score(clone(model), K, y, scorer=make_scorer(accuracy_score), train=train_index, test=test_index, verbose=0, parameters=p, fit_params=None)
                split_results.append(sc.get('test_scores'))
                params.append({'K_idx': K_idx, 'params': p})
        results.append(split_results)
    results = np.array(results)
    fin_results = results.mean(axis=0)
    best_idx = np.argmax(fin_results)
    ret_model = clone(model).set_params(**params[best_idx]['params'])
    return ret_model.fit(precomputed_kernels[params[best_idx]['K_idx']], y), params[best_idx]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ENZYMES',
                        help='name of dataset (default: ENZYMES)')
    parser.add_argument('--depth', type=int, default=2,
                        help='depth of bfs trees')
    parser.add_argument('--t', type=int, default=50,
                        help='number of partitionings')
    parser.add_argument('--c', type=int, default=4,
                        help='dim for pk')
    parser.add_argument('--tokenizer_file', type=str, default='./tokenizer/ENZYMES/depth_7_tokenizer.json',
                        help='pretrained tokenizer file')
    parser.add_argument('--batch_size', type=int, default=-1,
                        help='batch size')
    parser.add_argument('--model_file', type=str, default='./model/ENZYMES/best_model.pkl',
                        help='pretrained model file')
    parser.add_argument('--load_embedding', default=False, action='store_true',
                        help='use saved shortest-path embeddings')
    parser.add_argument('--C', type=float, default=100,
                        help='parameter C in SVM')
    parser.add_argument('--crossvalidation', default=True, action='store_false',
                        help='enable 10 fold cross validation')
    parser.add_argument('--gridsearch', default=False, action='store_true',
                        help='enable grid search')
    
    args = parser.parse_args()
    
    logging.info(f'dataset={args.dataset}, k={args.depth}')
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    mat_data = sio.loadmat(f'./data/{args.dataset}.mat')
    graph_data = mat_data['graph']
    graph_num = len(graph_data[0])
    
    graphs = {gidx: graph_data[0][gidx]['am'] for gidx in range(graph_num)}
    
    if not args.load_embedding:
        tokenizer = Tokenizer.from_file(args.tokenizer_file)
        model = torch.load(args.model_file, map_location='cpu')
        model.eval()
    
    time_start = time.time()
    
    node_embeddings = []
    for k in range(args.depth):
        if args.load_embedding:
            node_embedding = pickle.load(open(f'./sp_embs/{args.dataset}/emb_{k+1}.pkl', 'rb'))
            node_embedding = np.array(node_embedding)
        else:
            corpus_files = [f'./corpus/{args.dataset}/sp_depth_{k+1}']
        
            all_data = MyDataset(corpus_files, tokenizer)
            
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
    logging.info(f'Shape of node features: {ms_node_embeddings.shape}')
    
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
    
    partitioning = 'iforest'
    # partitioning = 'kmeans'
    
    c = args.c if partitioning == 'iforest' else int(args.c * graph_num)
    
    logging.info(f't={args.t}, c={c}')
    np.random.seed(42)
    
    if partitioning == 'iforest':
        partitions = iforest_partitioning(ms_graph_embeddings, args.t, c)
    else:
        partitions = kmeans_partitioning(ms_graph_embeddings, args.t, c)
    partition_indexs = get_partition_index(partitions, ms_graph_embeddings)
    pk_vectors = get_pk_vectors(partition_indexs, c)
    graph_features = np.array([np.mean(c, axis=0) for c in pk_vectors])
    
    kernel_matrix = graph_features @ graph_features.T
    diagonal_elements = np.diag(kernel_matrix)
    normalization_factors = np.sqrt(np.outer(diagonal_elements, diagonal_elements))
    kernel_matrix = kernel_matrix / normalization_factors
    
    time_end = time.time()
    
    logging.info(f'Kernel matrix is computed in {round(time_end - time_start, 2)}s')
    
    y = np.array([int(label) for label in mat_data['label']])
    
    param_grid = [{
        'C': np.logspace(0, 3, 4)
    }]
    gammas = [1]
    kernel_matrices = []
    for ga in gammas:
        kernel_matrices.append(kernel_matrix * ga)
    
    kernel_accuracy_scores = []

    cv = StratifiedKFold(n_splits=10, shuffle=True)
    
    best_C = []
    best_gamma = []
    for train_index, test_index in cv.split(kernel_matrices[0], y):
        K_train = [K[train_index][:, train_index] for K in kernel_matrices]
        K_test = [K[test_index][:, train_index] for K in kernel_matrices]
        y_train, y_test = y[train_index], y[test_index]

        if args.gridsearch:
            gs, best_params = grid_search(SVC(kernel='precomputed'), param_grid, K_train, y_train, cv=5)
            C_ = best_params['params']['C']
            gamma_ = gammas[best_params['K_idx']]
            y_pred = gs.predict(K_test[best_params['K_idx']])
        else:
            gs = SVC(C=args.C, kernel='precomputed').fit(K_train[0], y_train)
            y_pred = gs.predict(K_test[0])
            gamma_, C_ = gammas[0], 100
        best_C.append(C_)
        best_gamma.append(gamma_)

        kernel_accuracy_scores.append(accuracy_score(y_test, y_pred))
        if not args.crossvalidation:
            break
        
    if args.crossvalidation:
        logging.info('Mean 10-fold accuracy: {:2.2f} +- {:2.2f} %'.format(
            np.mean(kernel_accuracy_scores) * 100,
            np.std(kernel_accuracy_scores) * 100))
    else:
        logging.info('Final accuracy: {:2.3f} %'.format(np.mean(kernel_accuracy_scores) * 100))
