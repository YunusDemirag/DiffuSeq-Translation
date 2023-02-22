import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from typing import NamedTuple, Callable
import numpy as np
from tqdm import tqdm

class JSONLDataset(Dataset):
    '''Dataset for loading JSONL files.'''
    class SentencePair(NamedTuple):
        src: str
        trg: str
    
    def __init__(self, path):
        '''Loads the JSONL file at the given path.
        The file should contain a list of JSON objects, one per line.
        Each object should have a src and trg field.'''
        print('Loading dataset from', path)
        import json
        self.name = path.split('/')[-1]
        with open(path) as file:
            self.data = [self.SentencePair(**json.loads(line)) for line in file]
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    def __iter__(self):
        return iter(self.data)

class TokenizedDataset(Dataset):
    '''Dataset for loading tokenized data.'''
    
    class SentencePair(NamedTuple):
        src: np.ndarray
        trg: np.ndarray
    
    def __init__(self, dataset: JSONLDataset, tokenizer: Tokenizer):
        '''Tokenizes the given dataset.'''
        self.tokenizer = tokenizer
        self.name = dataset.name
        loading_bar = tqdm(dataset, f"Tokenizing Dataset {self.name}", leave=True)
        self.data = [self.SentencePair(
            src=np.array(tokenizer.encode(pair.src).ids),
            trg=np.array(tokenizer.encode(pair.trg).ids)
        ) for pair in loading_bar]
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    def __iter__(self):
        return iter(self.data)

class MaskedDataset(Dataset):
    '''Dataset for loading tokenized data.'''
    class SequenceWithMask(NamedTuple):
        seq: torch.Tensor
        mask: torch.Tensor
        labels: torch.Tensor
    
    def __init__(self, dataset: TokenizedDataset, prepare_and_mask: "Callable[[TokenizedDataset.SentencePair], list[SequenceWithMask]]"):
        '''Maskes and pepares the given dataset.'''
        self.name = dataset.name
        loading_bar = tqdm(dataset, f"Masking Dataset {self.name}", leave=True)
        self.data = [item for pair in loading_bar for item in prepare_and_mask(pair)]
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return { 'ids' : self.data[idx].seq, 'mask' : self.data[idx].mask, 'labels' : self.data[idx].labels }
