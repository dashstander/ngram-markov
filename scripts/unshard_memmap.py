import os
import argparse

import numpy as np
from tqdm import tqdm


def write_shard(full_dataset, pos, shard_filename, shard_size):  
    chunk_size = 1024 * 1024 * 1024 # 1 GB chunks
    shard_memmap = np.memmap(shard_filename, mode="r", order="C")
    for i in range(0, shard_size, chunk_size):
        chunk_end = min(i + chunk_size, shard_size)
        full_dataset[pos + i: pos + chunk_end] = shard_memmap[i:chunk_end]
    del shard_memmap


def unshard(
    input_file: str,
    num_shards: int,
    output_dir: str,
):
    """Reconstruct a Megatron .bin file from shards""" 

    input_dir = os.path.dirname(input_file)
    base_filename = os.path.basename(input_file)[:-19] # remove 00000-of-xxxxx.bin suffix from shard 0's filename

    # check size of non-final shard
    shard_filename = shard_filename = os.path.join(input_dir, base_filename) + f"-00000-of-{(num_shards - 1):05}.bin"
    shard_memmap = np.memmap(shard_filename, mode="r", order="C")
    SHARD_SIZE = shard_memmap.shape[0]

    # check size of final shard
    shard_filename = os.path.join(input_dir, base_filename) + f"-{(num_shards - 1):05}-of-{(num_shards - 1):05}.bin"
    shard_memmap = np.memmap(shard_filename, mode="r", order="C")
    final_shard_size = shard_memmap.shape[0]
    del shard_memmap

    # create full .bin file of proper size
    open(os.path.join(output_dir, base_filename) + ".bin", "w+").close()
    full_idx_map = np.memmap(os.path.join(output_dir, base_filename) + ".bin", shape=(SHARD_SIZE * (num_shards - 1) + final_shard_size), mode="w+", order="C")

    # chunk by iterating over file
    print(f"Loading {num_shards} shards from {input_dir}")
    for i in tqdm(range(num_shards)):
        
        shard_filename = os.path.join(input_dir, base_filename) + f"-{i:05}-of-{(num_shards - 1):05}.bin"
        print(shard_filename)
        #shard_memmap = np.memmap(shard_filename, mode="r", order="C")
        size = SHARD_SIZE if not (i == num_shards - 1) else final_shard_size
        buffer_bos = i * SHARD_SIZE
        write_shard(full_idx_map, buffer_bos, shard_filename, size)
        full_idx_map.flush()
        #full_idx_map[i * SHARD_SIZE: (i * SHARD_SIZE) + size] = shard_memmap

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Shard a single Megatron data .bin file"
    )

    ## CLI args
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to shard 0",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        help="Provide number of shards (The total seen in shard filenames + 1)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Folder to save .bin file into",
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    unshard(args.input_file, args.num_shards, args.output_dir)
