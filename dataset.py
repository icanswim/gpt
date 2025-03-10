import sys # required for relative imports in jupyter lab
sys.path.insert(0, '../')

import os
import requests
import tiktoken
import numpy as np

from sys import getsizeof

from cosmosis.dataset import CDataset


class TinyShakes(CDataset):
    """
    https://github.com/karpathy/nanoGPT
    """
    def __getitem__(self, i):
        
        X = self.ds[i:i+self.d_seq].astype(np.int64)
        y = self.ds[i+1:i+1+self.d_seq].astype(np.int64)
        pos = np.arange(0, self.d_seq, dtype=np.int64) 
        
        _data = {'tokens': X, 'y': y, 'position': pos}
        data = {}
        
        for feature, Transforms in self.transforms.items():
            out = _data[feature]
            for T in Transforms:
                out = T(out)
            data[feature] = out
            
        del _data
        return data

    def prompt(self, prompt):
        # encode the prompt with tiktoken gpt2 bpe
        tokens = self.encoding.encode_ordinary(prompt)
        ds = np.array(tokens, dtype=np.uint16)
        self.d_seq = ds.shape[-1]
        return ds
             
    def load_data(self, d_seq=1, n=338035, prompt=None):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        self.encoding = tiktoken.get_encoding("gpt2")
        self.d_seq, self.n = d_seq, n
        tiny_bin = './data/tinyshakes_stripped_encoded.bin'

        if prompt == None:
            if not os.path.exists('./data/tinyshakes.txt'):
                with open('./data/tinyshakes.txt', 'w', encoding='utf-8') as f:
                    f.write(requests.get(data_url).text)
                print('tinyshakes.txt downloaded and saved in ../gpt/data/')
            else:
                print('tinyshakes.txt loaded from saved file in ../gpt/data/')
    
            if not os.path.exists(tiny_bin):
                with open('./data/tinyshakes.txt', 'r', encoding='utf-8') as f:
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
            if n != 338035: 
                 ds_idx = list(np.random.choice(ds_idx, size=n, replace=False))
            ds.flush()
            self.ds_idx = ds_idx
        else:
            ds = self.prompt(prompt)
            self.ds_idx = [0]

        print('len(self.ds_idx): ', len(self.ds_idx))
        print('data.nbytes: ', ds.nbytes)
        return ds
        