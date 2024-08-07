from pathlib import Path
from tqdm import tqdm
from tokengrams import MemmapIndex


shards = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]



def build_and_write_shard(idx, shard_dir, index_dir):
    tokens_path = shard_dir / f'document-{idx:05}-of-00020.bin'
    index_path = index_dir / f'suffix_array_{idx:02}.idx'
    index = MemmapIndex.build(
        str(tokens_path),
        str(index_path),
        verbose=True
    )


if __name__ == '__main__':
    
    tokens_path = Path('/mnt/ssd-1/pile-tokens-ngrams/')
    index_dir = Path('/mnt/ssd-1/pile-suffix-arrays/')
    index_dir.mkdir(parents=True, exist_ok=True)
    #index_path = index_dir / 'suffix_tree.idx'
    print(f'Writing suffix trees to {index_dir}')
    for shard_idx in shards:
        build_and_write_shard(shard_idx, tokens_path, index_dir)