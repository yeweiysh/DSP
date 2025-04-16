from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
import argparse
import os

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ENZYMES',
                        help='name of dataset (default: ENZYMES)')
    parser.add_argument('--depth', type=int, default=7,
                        help='depth of bfs trees')
    args = parser.parse_args()
    
    corpus_cmd = f'python corpus.py --dataset {args.dataset} --depth {args.depth}'
    os.system(corpus_cmd)
    files = [f'./corpus/{args.dataset}/sp_depth_{args.depth}']
    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    trainer = BpeTrainer(special_tokens=['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]'])
    tokenizer.pre_tokenizer = WhitespaceSplit()
    
    tokenizer.train(files, trainer)
    
    save_dir = f'./tokenizer/{args.dataset}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tokenizer.save(f'{save_dir}/depth_{args.depth}_tokenizer.json')