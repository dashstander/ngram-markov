import os
import argparse
import numpy as np
from tqdm import tqdm

NEW_SHARD_SIZE = 2049 * 7_000_000

def read_shard(filename):
    return np.fromfile(filename, dtype=np.uint16)

def write_shard(data, filename):
    with open(filename, 'wb') as f:
        data.tofile(f)

def reshard(
    input_dir: str,
    output_dir: str,
):
    """Re-shard Megatron .bin files into new shards with NEW_SHARD_SIZE elements each"""

    # Get all input shard files
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.bin')])
    
    if not input_files:
        raise ValueError(f"No .bin files found in {input_dir}")

    # Extract base filename from the first shard
    base_filename = input_files[0].split('-')[0]

    # Calculate total number of elements
    total_elements = sum(os.path.getsize(os.path.join(input_dir, f)) // 2 for f in input_files)
    
    # Calculate number of new shards
    num_new_shards = (total_elements + NEW_SHARD_SIZE - 1) // NEW_SHARD_SIZE

    print(f"Re-sharding files from {input_dir} into {num_new_shards} new shards")

    current_shard = 0
    shard_pos = 0
    output_data = np.empty(NEW_SHARD_SIZE, dtype=np.uint16)

    for input_file in tqdm(input_files, desc="Processing input shards"):
        input_data = read_shard(os.path.join(input_dir, input_file))
        
        i = 0
        while i < len(input_data):
            space_left = NEW_SHARD_SIZE - shard_pos
            if space_left == 0:
                # Write full shard
                shard_filename = os.path.join(output_dir, f"{base_filename}-{current_shard:05d}-of-{num_new_shards-1:05d}.bin")
                write_shard(output_data, shard_filename)
                current_shard += 1
                shard_pos = 0
                space_left = NEW_SHARD_SIZE

            # Copy data
            elements_to_copy = min(space_left, len(input_data) - i)
            output_data[shard_pos:shard_pos+elements_to_copy] = input_data[i:i+elements_to_copy]
            shard_pos += elements_to_copy
            i += elements_to_copy

    # Write last shard if there's any data left
    if shard_pos > 0:
        shard_filename = os.path.join(output_dir, f"{base_filename}-{current_shard:05d}-of-{num_new_shards-1:05d}.bin")
        write_shard(output_data[:shard_pos], shard_filename)

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

    reshard(args.input_dir, args.output_dir)