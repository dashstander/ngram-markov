from argparse import ArgumentParser
from copy import deepcopy
from functools import partial
from ngram_markov.sparse import IncrementalCSRMatrix
import numpy as np
from pathlib import Path
import scipy.sparse as sp
from tokengrams import MemmapIndex


parser = ArgumentParser()
parser.add_argument('--suffix-tree-path', type=str, required=True)
parser.add_argument('--tokens-path', type=str, required=True)
parser.add_argument('--ngram', type=int, default=2)
parser.add_argument('--num-tokens', type=int, required=True)


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


def get_index_queries(prev_queries, prev_inds, counts):
    queries = []
    for i, query in zip(prev_inds, prev_queries):
        _, nonzero_cols = counts[[i], :].nonzero()
        queries.extend([
            [deepcopy(query) + j] for j in nonzero_cols.tolist()
        ])
    return queries

def query_to_index(num_tokens, query):
    exponents = reversed(list(range(len(query))))
    base_token = [num_tokens ** i for i in exponents]
    return sum([q * b for q, b, in zip(query, base_token)])


def get_ngram_counts(suffix_tree, prev_counts, n, num_tokens):
    index_to_tokens_fn = partial(int_to_base_x, num_tokens, n - 2)
    q2i_fn = partial(query_to_index, num_tokens)
    count_matrix = IncrementalCSRMatrix((num_tokens**(n-1), num_tokens), np.float32)

    nonzero_prev_inds = np.argwhere(prev_counts.sum(axis=1) != 0).squeeze().tolist()
    nonzero_prev_grams = [index_to_tokens_fn(i) for i in nonzero_prev_inds]
    queries = get_index_queries(nonzero_prev_grams, nonzero_prev_inds, prev_counts)
    raw_ngram_counts = suffix_tree.batch_count_next(queries, num_tokens - 1)
    for query, ngram_row in zip(queries, raw_ngram_counts):
        idx = q2i_fn(query)
        count_matrix.append_row(np.array(ngram_row, dtype=np.float32), idx)
    return count_matrix.tocsr()


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
    np.savez(data_dir / '1grams.npz', unigram_counts)
    np.savez(data_dir / '2grams.npz', bigram_counts)
    curr_mat = bigram_counts
    curr_n = 2
    while curr_n < max_n:
        curr_n += 1
        print(f'Querying for {curr_n}-gram counts...')
        curr_mat = get_ngram_counts(curr_mat, curr_n, num_tokens)
        print(f'Writing {curr_n}-gram matrix...')
        np.savez(data_dir / f'{curr_n}grams.npz', curr_mat)


if __name__ == '__main__':
    args, _ = parser.parse_known_args()
    main(args)
