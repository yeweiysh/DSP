import argparse
import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import logging

from tokenizers import Tokenizer
from data import MyDataset
from model import SP_Transformer


logging.basicConfig(
    format='%(message)s',
    level=logging.DEBUG,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ENZYMES',
                        help='name of dataset (default=ENZYMES)')
    parser.add_argument('--depth', type=int, default=7,
                        help='depth of bfs trees')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='iterations of training')
    parser.add_argument('--d_model', type=int, default=16,
                        help='the number of expected features in the encoder/decoder inputs (default=512)')
    parser.add_argument('--nhead', type=int, default=4,
                        help='the number of heads in the multiheadattention models (default=8)')
    parser.add_argument('--num_encoder_layers', type=int, default=3,
                        help='the number of sub-encoder-layers in the encoder (default=6)')
    parser.add_argument('--num_decoder_layers', type=int, default=3,
                        help='the number of sub-decoder-layers in the decoder (default=6)')
    parser.add_argument('--dim_feedforward', type=int, default=256,
                        help='the dimension of the feedforward network model (default=2048)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='the dropout value (default=0.1)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='the learning rate value (default=0.0005)')
    parser.add_argument('--use_gpu', default=True, action='store_false',
                        help='enable gpu training')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='id of gpu device')
    parser.add_argument('--save_model', default=True, action='store_false',
                        help='whether to save model or not')
    args = parser.parse_args()
    
    logging.info(f'dataset={args.dataset}, d={args.depth}')
    
    if args.use_gpu:
        logging.info(f'gpu id is {args.gpu_id}')
        device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    else:
        logging.info('use cpu')
        device = torch.device('cpu')
    
    corpus_files = [f'./corpus/{args.dataset}/sp_depth_{args.depth}']
    
    tokenizer_dir = f'./tokenizer/{args.dataset}/depth_{args.depth}_tokenizer.json'
    tokenizer = Tokenizer.from_file(tokenizer_dir)
    with open(tokenizer_dir, 'r') as f:
        data = json.load(f)
    vocab_size = len(data['model']['vocab'])
    
    all_data = MyDataset(corpus_files, tokenizer)
    
    if args.batch_size == -1:
        # full batch
        logging.info('full batch training')
        my_loader = DataLoader(dataset=all_data, batch_size=len(all_data), shuffle=True)
    else:
        # mini batch
        logging.info(f'mini batch training, batch_size={args.batch_size}')
        my_loader = DataLoader(dataset=all_data, batch_size=args.batch_size, shuffle=True)
        
    logging.info(f'vocab_size={vocab_size}, d_model={args.d_model}, nhead={args.nhead}, num_encoder_layers={args.num_encoder_layers}, num_decoder_layers={args.num_decoder_layers}, dim_feedforward={args.dim_feedforward}, dropout={args.dropout}')
    
    model = SP_Transformer(
        vocab_size=vocab_size,
        max_len=len(all_data[0]),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)
    
    logging.info(f'lr={args.lr}, epochs={args.epochs}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # epochs = 200
    for epoch in range(args.epochs):
        model.train()
        t_start = time.time()
        total_loss = 0
        cnt = 0
        for src in my_loader:
            cnt += 1
            tgt = src[:, :-1]
            y = src[:, 1:]
            optimizer.zero_grad()
            output = model(src.to(device), tgt.to(device))
            output = model.predictor(output)
            loss = criterion(output.contiguous().view(-1, output.size(-1)), y.to(device).contiguous().view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        t_end = time.time()
        loss_train = total_loss / cnt
        print(
            'Epoch: {:04d}'.format(epoch + 1),
            'loss_train: {:.4f}'.format(loss_train),
            'time: {:.4f}s'.format(t_end - t_start)
        )
        
    if args.save_model:
        save_dir = f'./model/{args.dataset}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
          
        model_dir = f'{save_dir}/best_model.pkl'
        
        torch.save(model, model_dir)
        logging.info(f'model is saved in {model_dir}')
