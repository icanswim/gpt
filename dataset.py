import sys # required for relative imports in jupyter lab
sys.path.insert(0, '../')

import os
import requests
import tiktoken
import numpy as np

from sys import getsizeof

from cosmosis.dataset import TDataset


class TinyShakes(TDataset):
    """
    https://github.com/karpathy/nanoGPT
    """      
    def load_data(self, dir='./data', d_seq=1, n=338035, prompt=None, tokenizer=tiktoken):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        self.encoding = tokenizer.get_encoding("gpt2")
        self.d_seq, self.n, self.dir = d_seq, n, dir
        tiny_bin = os.path.join(self.dir, 'tinyshakes_stripped_encoded.bin')

        if prompt == None:
            if not os.path.exists(os.path.join(self.dir, 'tinyshakes.txt')):
                with open(os.path.join(self.dir, 'tinyshakes.txt'), 'w', encoding='utf-8') as f:
                    f.write(requests.get(data_url).text)
                print('tinyshakes.txt downloaded and saved in ../gpt/data/')
            else:
                print('tinyshakes.txt loaded from saved file in ../gpt/data/')
    
            if not os.path.exists(tiny_bin):
                with open(os.path.join(self.dir, 'tinyshakes.txt'), 'r', encoding='utf-8') as f:
                    data = f.read()
                    data = data.replace('\n', ' ')
                # encode with tiktoken gpt2 bpe
                tokens = self.encoding.encode_ordinary(data)
                tokens = np.array(tokens, dtype=np.uint16)
                tokens.tofile(tiny_bin)
                print('text has been tokenized and saved in file {}'.format(tiny_bin))
            else:
                print('tokens loaded from file {}'.format(tiny_bin))
                
            ds = np.memmap(tiny_bin, dtype=np.uint16, mode='r')
            ds_idx = list(range(ds.shape[-1]-self.d_seq)) # 338035
            ds.flush()
            self.ds_idx = ds_idx
        else:
            ds = self.encoding.encode_ordinary(prompt)
            ds = np.array(ds, dtype=np.uint16)
            self.ds_idx = [0]
            self.d_seq = ds.shape[0] - 2

        print('len(ds): ', len(ds))
        return ds.copy()
    
    def prompt(self, prompt):
        ds = self.encoding.encode_ordinary(prompt)
        ds = np.array(ds, dtype=np.uint16)
        self.ds_idx = [0]
        self.d_seq = ds.shape[0] - 2
        return ds.copy()
        