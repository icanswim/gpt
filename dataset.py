import sys # required for relative imports in jupyter lab
sys.path.insert(0, '../')

import os
import requests
import tiktoken
import numpy as np

from cosmosis.dataset import CDataset


class TinyShakes(CDataset):
    """
    https://github.com/karpathy/nanoGPT
    """

    def __getitem__(self, i):         
        ix = np.random.randint(low=0, high=len(self.ds) - self.block_size, size=(self.batch_size,))
        X1 = np.stack([(self.ds[i:i+self.block_size]).astype(np.int64) for i in ix])
        X2 = np.stack([(self.ds[i+1:i+1+self.block_size]).astype(np.int64) for i in ix])
        
        _data = {'X1': X1, 'X2': X2}
        data = {}
        
        for feature, Transforms in self.transforms.items():
            out = _data[feature]
            for T in Transforms:
                out = T(out)
            data[feature] = out
            
        del _data
        return data
                
    def load_data(self, block_size, batch_size):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        self.enc = tiktoken.get_encoding("gpt2")
        self.ds_idx = []
        self.block_size = block_size
        self.batch_size = batch_size
        
        if not os.path.exists('./data/tinyshakes.txt'):
            with open('./data/tinyshakes.txt', 'w', encoding='utf-8') as f:
                f.write(requests.get(data_url).text)
            print('tinyshakes.txt downloaded and saved in ../gpt/data/')
        else:
            print('tinyshakes.txt loaded from saved file in ../gpt/data/')

        if not os.path.exists('./data/tinyshakes_encoded.bin'):
            with open('./data/tinyshakes.txt', 'r', encoding='utf-8') as f:
                data = f.read()
            # encode with tiktoken gpt2 bpe
            tokens = self.enc.encode_ordinary(data)
            tokens = np.array(tokens, dtype=np.uint16)
            tokens.tofile('./data/tinyshakes_encoded.bin')
            print('text has been tokenized and saved in file ./data/tinyshakes_encoded.bin')
        else:
            print('tokens loaded from file ./data/tinyskakes_encoded.bin')
            
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/\
        # numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        data = np.memmap('./data/tinyshakes_encoded.bin', dtype=np.uint16, mode='r')
        return data