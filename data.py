import torch
from torch.utils.data import Dataset
from tokenizers.processors import TemplateProcessing


class MyDataset(Dataset):
    def __init__(self, files, tokenizer):
        super(MyDataset, self).__init__()
        self.corpus = []
        for file in files:
            with open(file, 'r') as f:
                self.corpus.extend(f.readlines())
        
        tokenizer.post_processor = TemplateProcessing(
            single='[CLS] $A [SEP]',
            special_tokens=[
                ('[CLS]', tokenizer.token_to_id('[CLS]')),
                ('[SEP]', tokenizer.token_to_id('[SEP]'))
            ]
        )
        
        tokenizer.enable_padding(
            pad_id=tokenizer.token_to_id('[PAD]'),
            pad_token='[PAD]'
        )
        
        self.token_infos = tokenizer.encode_batch(self.corpus)
        
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, index):
        return torch.LongTensor(self.token_infos[index].ids)