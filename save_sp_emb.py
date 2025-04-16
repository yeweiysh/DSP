import argparse
import torch
import time
import scipy.io as sio
import numpy as np
from tokenizers import Tokenizer
from data import MyDataset
from torch.utils.data import DataLoader
import os
import pickle
import logging


logging.basicConfig(
    format='%(message)s',
    level=logging.DEBUG,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ENZYMES',
                        help='name of dataset (default: ENZYMES)')
    parser.add_argument('--depth', type=int, default=7,
                        help='depth of bfs trees')
    parser.add_argument('--tokenizer_file', type=str, default='./tokenizer/ENZYMES/depth_7_tokenizer.json',
                        help='pretrained tokenizer file')
    parser.add_argument('--batch_size', type=int, default=-1,
                        help='batch size')
    parser.add_argument('--model_file', type=str, default='./model/ENZYMES/best_model.pkl',
                        help='pretrained model file')
    
    args = parser.parse_args()
    
    logging.info(f'dataset={args.dataset}, d={args.depth}')
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    mat_data = sio.loadmat(f'./data/{args.dataset}.mat')
    graph_data = mat_data['graph']
    graph_num = len(graph_data[0])
    
    graphs = {gidx: graph_data[0][gidx]['am'] for gidx in range(graph_num)}
    
    tokenizer = Tokenizer.from_file(args.tokenizer_file)
    
    model = torch.load(args.model_file, map_location='cpu')
    model.eval()
    
    time_start = time.time()
    
    for d in range(args.depth):
        corpus_cmd = f'python corpus.py --dataset {args.dataset} --depth {d+1}'
        os.system(corpus_cmd)
        
        corpus_files = [f'./corpus/{args.dataset}/sp_depth_{d+1}']
    
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
            
        save_dir = f'./sp_embs/{args.dataset}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        emb_dir = f'{save_dir}/emb_{d+1}.pkl'
        with open(emb_dir, 'wb') as f:
            pickle.dump(outs, f)
    
    time_end = time.time()
    
    logging.info(f'Saved in {round(time_end - time_start, 2)}s')
