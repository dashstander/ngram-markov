from argparse import ArgumentParser
from copy import deepcopy
from functools import partial
from ngram_markov.sparse import IncrementalCSRMatrix, IncrementalCOOMatrix
import numpy as np
from pathlib import Path
import scipy.sparse as sp
from tokengrams import MemmapIndex
from tqdm import tqdm


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


parser = ArgumentParser()
parser.add_argument('--suffix-tree-path', type=str, required=True)
parser.add_argument('--tokens-path', type=str, required=True)
parser.add_argument('--ngram', type=int, default=2)
parser.add_argument('--num-tokens', type=int, required=True)


def batch_n(x, n):
    return [x[i:i+n] for i in range(0, len(x), n)]

"""
def int_to_base_x(base: int, minlen: int, x: int):
    curr_val, mod = divmod(x, base)
    nums = [mod]
    while True:
        if curr_val < base:
            nums.insert(0, curr_val)
            break
        else:
            curr_val, mod = divmod(curr_val, base)
            nums.insert(0, mod)
    need_extras = minlen - len(nums)
    if need_extras > 0:
        extras = [0] * need_extras
        nums = extras + nums
    return nums
"""

def int_to_base_x(base: int, minlen: int, arr: np.ndarray):
    result = np.zeros((len(arr), minlen), dtype=int)
    arr_copy = arr.copy()

    for i in range(minlen - 1, -1, -1):
        result[:, i] = arr_copy % base
        arr_copy //= base

    return result



def get_index_queries(prev_queries, prev_inds, counts):
    queries = []
    for i, query in zip(prev_inds, prev_queries):
        _, nonzero_cols = counts.getrow(i).nonzero()
        queries.extend([
            deepcopy(query) + [j] for j in nonzero_cols.tolist()
        ])
    return queries


def query_to_index(num_tokens, query):
    exponents = reversed(list(range(len(query))))
    base_token = [num_tokens ** i for i in exponents]
    return sum([q * b for q, b, in zip(query, base_token)])



def append_row(matrix, row, idx):
    nonzero_indices = np.nonzero(row)[0]
    for j in nonzero_indices:
        matrix[idx, j] = row[j]

def _add_rows_to_tree(suffix_tree, queries, num_tokens, matrix):
    q2i_fn = partial(query_to_index, num_tokens)
    #print('Querying suffix tree')
    raw_ngram_counts = suffix_tree.batch_count_next(queries, num_tokens - 1)
    #print('Adding rows')
    for query, ngram_row in zip(queries, raw_ngram_counts):
        idx = q2i_fn(query)
        append_row(matrix, np.array(ngram_row, dtype=np.float64), idx)


def get_ngram_counts(suffix_tree, prev_counts, n, num_tokens):
    #prev_counts = prev_counts.todok()
    index_to_tokens_fn = partial(int_to_base_x, num_tokens, n - 2)
    #count_matrix = IncrementalCOOMatrix((num_tokens**(n-1), num_tokens), np.float64, np.int64)
    count_matrix = sp.dok_array((num_tokens**(n-1), num_tokens), dtype=np.float64)
    nonzero_rows, nonzero_cols = prev_counts.nonzero()
    nonzero_prev_grams = index_to_tokens_fn(nonzero_rows).tolist()
    queries = [row + col for row, col in zip(nonzero_prev_grams, nonzero_cols[:, None].tolist())]
    print(f'{len(queries)} queries')
    for query_batch in tqdm(batch_n(queries, 30_000)):
        _add_rows_to_tree(suffix_tree, query_batch, num_tokens, count_matrix)
    return count_matrix.tocoo()


def main(args):
    max_n = args.ngram
    data_dir = Path(args.suffix_tree_path)
    suffix_index = MemmapIndex(args.tokens_path, str(data_dir / 'suffix_tree.idx'))

    num_tokens = args.num_tokens
    max_token = num_tokens - 1

    bigram_queries = [[i] for i in range(num_tokens)]
    print('Querying unigram and bigram statistics....')
    bigram_counts = np.array(
        suffix_index.batch_count_next(bigram_queries, max_token),
        dtype=np.float32
    )
    unigram_counts = bigram_counts.sum(axis=0)
    bigram_counts = sp.coo_array(bigram_counts)
    np.save(data_dir / '1grams.npy', unigram_counts)
    sp.save_npz(data_dir / '2grams.npz', bigram_counts)
    curr_mat = bigram_counts
    curr_n = 2
    while curr_n < max_n:
        curr_n += 1
        print(f'Querying for {curr_n}-gram counts...')
        curr_mat = get_ngram_counts(suffix_index, curr_mat, curr_n, num_tokens)
        print(f'Writing {curr_n}-gram matrix...')
        sp.save_npz(data_dir / f'{curr_n}grams.npz', curr_mat)


if __name__ == '__main__':
    args, _ = parser.parse_known_args()
    main(args)
