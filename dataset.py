import sys # required for relative imports in jupyter lab
sys.path.insert(0, '../')

import os
import requests
import tiktoken
import numpy as np

from torch import stack, from_numpy, randint

from cosmosis.dataset import CDataset


class TinyShakes(CDataset):

    def __getitem__(self, i):
        block_size = 5
        batch_size = 3

        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/\
        # numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    
        data = np.memmap('./data/encoded.bin', dtype=np.uint16, mode='r')
    
        ix = randint(len(data) - block_size, (batch_size,))
        
        x = stack([from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = stack([from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        
        return x, y

    def load_data(self):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        self.enc = tiktoken.get_encoding("gpt2")
        self.ds_idx = []
        
        if not os.path.exists('./data/tinyshakespeare.txt'):
            with open('./data/tinyshakespeare.txt', 'w', encoding='utf-8') as f:
                f.write(requests.get(data_url).text)
            print('tinyshakespeare.txt saved in ../gpt/data/')

        if not os.path.exists('./data/encoded.bin'):
            with open('./data/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
                data = f.read()
                print('len(data): ', len(data))
                
            # encode with tiktoken gpt2 bpe
            tokens = self.enc.encode_ordinary(data)
            tokens = np.array(tokens, dtype=np.uint16)
            tokens.tofile('./data/'+'encoded.bin')
            print('tokens created and saved in file ./data/encoded.bin')
        else:
            print('tokens loaded from file ./data/encoded.bin')
