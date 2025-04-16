import argparse
import logging
import re
import os
import time
from tqdm import trange
import scipy.io as sio
import networkx as nx
import numpy as np
from collections import deque
from operator import itemgetter


logging.basicConfig(
    format='%(message)s',
    level=logging.DEBUG,
)


def generic_bfs_edges(G, label, source, neighbors=None, depth_limit=None):
    visited = {source}
    if depth_limit is None:
        depth_limit = len(G)

    neigh = list(neighbors(source))

    neighbor_list = []
    for nei in neigh:
        neighbor_dict = {}
        neighbor_dict['vertex'] = int(nei)
        neighbor_dict['label'] = label[int(nei)]
        nbrs = list(neighbors(nei))
        tree = ''
        for nbr in nbrs:
            if nbr not in visited and nbr not in neigh:
                tree += str(label[int(nbr)])
        neighbor_dict['children'] = sorted(tree)
        neighbor_list.append(neighbor_dict)

    neighbor_list_sorted = sorted(neighbor_list, key=itemgetter('label', 'children'))
    sortedneighbor = []
    for neighbor_dict_sorted in neighbor_list_sorted:
        vertex = neighbor_dict_sorted['vertex']
        sortedneighbor.append(vertex)

    queue = deque([(source, depth_limit, iter(sortedneighbor))])

    while queue:
        parent, depth_now, children = queue[0]
        try:
            child = next(children)
            if child not in visited:
                yield parent, child
                visited.add(child)
                if depth_now > 1:
                    chil = list(neighbors(child))
                    child_list = []

                    for chi in chil:
                        child_dict = {}
                        child_dict['vertex'] = int(chi)
                        child_dict['label'] = label[int(chi)]
                        nbrs = list(neighbors(chi))
                        tree = ''
                        for nbr in nbrs:
                            if nbr not in visited and nbr not in chil:
                                tree += str(label[int(nbr)])
                        child_dict['children'] = sorted(tree)
                        child_list.append(child_dict)

                    child_list_sorted = sorted(child_list, key=itemgetter('label', 'children'))
                    sortedchild = []
                    for child_dict_sorted in child_list_sorted:
                        vertex = child_dict_sorted['vertex']
                        sortedchild.append(vertex)

                    queue.append((child, depth_now - 1, iter(sortedchild)))
        except StopIteration:
            queue.popleft()


def bfs_edges(G, label, source, reverse=False, depth_limit=None):
    if reverse and G.is_directed():
        successors = G.predecessors
    else:
        successors = G.neighbors
    # TODO In Python 3.3+, this should be `yield from ...`
    for e in generic_bfs_edges(G, label, source, successors, depth_limit):
        yield e
        
        
def compute_centrality(adj_matrix):
    n = len(adj_matrix)
    adj_matrix = adj_matrix + np.eye(n)
    ec = np.zeros(n)
    nxgraph = nx.from_numpy_matrix(adj_matrix)
    nodes = nx.eigenvector_centrality(nxgraph, max_iter=1000, tol=1.0e-4)
    for i in range(len(nodes)):
        ec[i] = nodes[i]

    return ec
                    
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ENZYMES',
                        help='name of dataset (default: ENZYMES)')
    parser.add_argument('--depth', type=int, default=7,
                        help='depth of bfs trees')
    args = parser.parse_args()
    
    logging.info(f'# Dataset: {args.dataset}')
    logging.info(f'# Depth of BFS trees: {args.depth}')
    
    mat_data = sio.loadmat(f'./data/{args.dataset}.mat')
    graph_data = mat_data['graph']
    graph_num = len(graph_data[0])
    
    corpus_root = f'./corpus/{args.dataset}'
    if not os.path.exists(corpus_root):
        os.makedirs(corpus_root)
    corpus_file = f'{corpus_root}/sp_depth_{args.depth}'
    if os.path.exists(corpus_file):
        logging.info('Target corpus already exists!')
    else:
        time_start = time.time()
        
        if args.dataset in ['IMDB-BINARY', 'IMDB-MULTI', 'DBLP_v1']:
            degree_as_label = True
            logging.info('Use node degree as node label!')
        else:
            degree_as_label = False
        
        with open(corpus_file, 'w') as f:
            for gidx in trange(graph_num):
                adj_matrix = graph_data[0][gidx]['am']
                
                eigen_cen = compute_centrality(adj_matrix)
                nxgraph = nx.from_numpy_matrix(adj_matrix)
                
                if degree_as_label:
                    label = [deg[1] for deg in nx.degree(nxgraph)]
                else:
                    label = graph_data[0][gidx]['nl'].T[0]
                
                judge_set = set()
                for node in range(len(nxgraph)):
                    paths_node = []
                    paths_node.append(str(node))
                    judge_set.add(str(node))
                    edges = list(bfs_edges(nxgraph, label, source=node, depth_limit=args.depth))
                    pathss = []
                    for u, v in edges:
                        path = list(nx.shortest_path(nxgraph, source=node, target=v))
                        strpath = ''
                        cnt = 0
                        for vertex in path:
                            strpath += str(vertex)
                            cnt += 1
                            if cnt < len(path):
                                strpath += ','
                        pathss.append(strpath)
                    for path in pathss:
                        vertices = re.split(',', path)
                        rvertices = list(reversed(vertices))
                        rpath = ''
                        cnt = 0
                        for rv in rvertices:
                            rpath += rv
                            cnt += 1
                            if cnt < len(rvertices):
                                rpath += ','
                        if rpath not in judge_set:
                            judge_set.add(path)
                            paths_node.append(path)
                        else:
                            paths_node.append(rpath)
                            
                    paths_node_dic = {}
                    for sp in paths_node:
                        nds = re.split(',', sp)
                        ecs = 0
                        for nd in nds:
                            ecs += eigen_cen[int(nd)]
                        if ecs not in paths_node_dic:
                            paths_node_dic[ecs] = [sp]
                        else:
                            paths_node_dic[ecs].append(sp)
                    sorted_paths_node = []
                    for ec in sorted(paths_node_dic):
                        sorted_paths_node.extend(paths_node_dic[ec])
                    
                    labeledpaths = []
                    for pathstr in sorted_paths_node:
                        labeledpath = ''
                        vertices = re.split(',', pathstr)
                        cnt = 0
                        for vertex in vertices:
                            labeledpath += str(int(label[int(vertex)]))
                            cnt += 1
                            if cnt < len(vertices):
                                labeledpath += ','
                        labeledpaths.append(labeledpath)
                    labeledpaths = ' '.join(labeledpaths)
                    
                    f.write(labeledpaths)
                    f.write('\n')
                
        time_end = time.time()
        logging.info(f'Time eclipses: {round(time_end - time_start, 2)}s.')
        logging.info(f'Corpus is saved in {corpus_file}.')