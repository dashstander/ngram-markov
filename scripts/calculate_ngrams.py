from ngram_markov.ngrams import create_ngrams
from random import randint
from tokengrams import ShardedMemmapIndex
from pathlib import Path
import numpy as np
from scipy import sparse
from tqdm import tqdm
import torch

ngram_values = range(2, 21)
vocab_size = 50_304

batch_size = 128
block_size = 512


def get_batch():
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    i = randint(0, 20)
    data = np.memmap(f'/mnt/ssd-1/pile-ngrams-tokens/document-{i:05}-of-00020.bin', dtype=np.uint16, mode='r')    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    return x


def get_ngram_counts(index, data, n):
    ngrams = create_ngrams(data.cpu(), n-1)
    raw_counts = index.batch_count_next(ngrams.reshape(-1, n-1).numpy())
    return sparse.csr_array(raw_counts, dtype=np.float32)


def main():
    num_batches = 21
    ngram_values = list(range(2, 21))
    data_dir = Path('/mnt/ssd-1/dashiell/ngram_data')

    sa_path = Path('/mnt/ssd-1/pile-suffix-arrays/')
    tokens_path = Path('/mnt/ssd-1/pile-ngrams-tokens')
    paths = [(str(sa_fp), str(t_fp)) for sa_fp, t_fp in zip(sorted(tokens_path.iterdir()), sorted(sa_path.glob('*.idx')))]

    index = ShardedMemmapIndex(paths, vocab=vocab_size)

    for i in range(1, num_batches):
        batch_dir = data_dir / f'{i}'
        batch_dir.mkdir(exist_ok=True, parents=True)
        data = get_batch()
        np.save(batch_dir / 'tokens.npy', data.numpy())
        for n in tqdm(ngram_values):
            print('#' * 30)
            counts = get_ngram_counts(index, data, n)
            sparse.save_npz(batch_dir / f'{n}.npz', counts)


if __name__ == '__main__':
    main()