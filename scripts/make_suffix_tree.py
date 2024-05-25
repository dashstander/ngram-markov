from pathlib import Path
from tokengrams import MemmapIndex
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--data-path', type=str, required=True)
parser.add_argument('--output-dir', type=str, required=True)


if __name__ == '__main__':
    args, _ = parser.parse_known_args()
    tokens_path = Path(args.data_path)
    index_dir = Path(args.output_path)
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / 'suffix_tree.idx'
    print(f'Writing suffix tree to {index_path}')
    index = MemmapIndex.build(
        tokens_path,
        index_path,
        verbose=True
    )
