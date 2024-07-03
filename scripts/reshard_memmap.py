import argparse
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm


NEW_SHARD_SIZE = 2049 * 7_000_000
CHUNK_SIZE = 1024 * 1024 * 1024 // 2  # 1 GB chunks (in uint16 elements)

def read_chunk(shard_memmap, start, size):
    chunk_end = min(start + size, shard_memmap.shape[0])
    return shard_memmap[start:chunk_end]


def write_chunk(shard_memmap, start, data):
    chunk_end = start + len(data)
    shard_memmap[start:chunk_end] = data


def get_size(filename):
    return np.memmap(filename, mode='r', dtype=np.uint16).shape


def reshard(
    input_dir: Path,
    output_dir: Path,
):
    """Re-shard Pile .bin files into new shards with NEW_SHARD_SIZE elements each"""

    # Get all input shard files
    input_files = sorted([f for f in input_dir.iterdir() if f.name.endswith('.bin')])
    
    if not input_files:
        raise ValueError(f"No .bin files found in {input_dir}")

    # Extract base filename from the first shard
    base_filename = input_files[0].name.split('-')[0]

    # Calculate total number of elements
    total_elements = sum([get_size(filename)[0] for filename in input_files])
    
    # Calculate number of new shards
    num_new_shards = (total_elements + NEW_SHARD_SIZE - 1) // NEW_SHARD_SIZE

    print(f"Re-sharding files from {input_dir} into {num_new_shards} new shards")

    current_shard = 0
    shard_pos = 0
    output_shard = None

    for input_file in tqdm(input_files, desc="Processing input shards"):
        input_memmap = np.memmap(os.path.join(input_dir, input_file), dtype=np.uint16, mode='r')
        
        for chunk_start in range(0, len(input_memmap), CHUNK_SIZE):
            input_chunk = read_chunk(input_memmap, chunk_start, CHUNK_SIZE)
            
            while len(input_chunk) > 0:
                if output_shard is None:
                    # Create a new output shard
                    shard_filename = output_dir / f"{base_filename}-{current_shard:05d}-of-{num_new_shards-1:05d}.bin"
                    output_shard = np.memmap(shard_filename, dtype=np.uint16, mode='w+', shape=(NEW_SHARD_SIZE,))
                    shard_pos = 0

                space_left = NEW_SHARD_SIZE - shard_pos
                chunk_to_write = input_chunk[:space_left]
                write_chunk(output_shard, shard_pos, chunk_to_write)
                shard_pos += len(chunk_to_write)
                input_chunk = input_chunk[space_left:]

                if shard_pos == NEW_SHARD_SIZE:
                    # Current shard is full, close it and move to the next
                    output_shard.flush()
                    del output_shard
                    output_shard = None
                    current_shard += 1

        del input_memmap

    # Close the last shard if it's not full
    if output_shard is not None:
        if shard_pos < NEW_SHARD_SIZE:
            # Resize the last shard to its actual size
            output_shard.flush()
            del output_shard
            last_shard_filename = output_dir / f"{base_filename}-{current_shard:05d}-of-{num_new_shards-1:05d}.bin"
            os.truncate(last_shard_filename, shard_pos * 2)  # *2 because uint16 is 2 bytes
        else:
            output_shard.flush()
            del output_shard

    print(f"Re-sharding complete. {num_new_shards} shards saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-shard Megatron data .bin files"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing input .bin files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save re-sharded .bin files",
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    reshard(Path(args.input_dir), Path(args.output_dir))
