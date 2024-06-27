from pathlib import Path
from tokengrams import MemmapIndex, InMemoryIndex
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--data-path', type=str, required=True)
parser.add_argument('--output-dir', type=str, required=True)
parser.add_argument('--in-memory', action='store_true', default=False)


if __name__ == '__main__':
    args, _ = parser.parse_known_args()
    tokens_path = Path(args.data_path)
    index_dir = Path(args.output_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / 'suffix_tree.idx'
    print(f'Writing suffix tree to {index_path}')
    if args.in_memory:
        index = InMemoryIndex.from_token_file(str(tokens_path), True)
        index.save(str(index_path))
    else:
        index = MemmapIndex.build(
            str(tokens_path),
            str(index_path),
            verbose=True
        )
