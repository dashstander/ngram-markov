from argparse import ArgumentParser
from huggingface_hub import HfApi
import numpy as np
from pathlib import Path
import polars as pl
from scipy import sparse
import torch
import tqdm
from transformers import GPTNeoXForCausalLM

from ngram_markov.ngrams import kl_divergence, 


parser = ArgumentParser()
parser.add_argument('--min-n', type=int, default=2)
parser.add_argument('--max-n', type=int, max=20)
parser.add_argument('--model-path', type=str, default='EleutherAI/pythia-70m-deduped')


def get_hf_revisions(api, hf_model_path: str):
    """Takes a HuggingFace model name and returns all of the 'revisions' of that model, e.g. 'step1', 'step200', etc...

    """   
    refs = api.list_repo_refs(hf_model_path)
    return sorted([branch.name for branch in refs.branches if branch.name != 'main'])


def kl_data_to_df(data, num_steps, n):
    batch_size = data.shape[0]
    df = (
        pl.DataFrame(data.numpy(), schema=[f'{i}' for i in range(batch_size)])
        .with_row_index()
        .rename({'index': 'position'})
        .unpivot(index='position', variable_name='sequence', value_name='kl')
        .with_columns([
            pl.col('sequence').str.to_integer(),
            pl.lit(num_steps).alias('steps'),
            pl.lit(n).alias('n')
        ])
        .select(pl.exclude('sequence'))
    )
    return df

@torch.no_grad()
def estimate_model_kl(model, n, data, counts):
    model.eval()
    ngram_dist = (counts / counts.sum(dim=1, keepdims=True)).reshape(data.shape[0], data.shape[1] - n + 2, vocab_size)
    log_ngrams = torch.log(ngram_dist + 1.0e-10)
    batch_size = 8
    all_logits = []
    for batch in data.split(batch_size):
        logits = model(batch.to('cuda'))['logits']
        all_logits.append(logits.to('cpu'))
    all_logits = torch.concatenate(all_logits, dim=0)
    logprobs = torch.log_softmax(all_logits[:, (n-2):, :], dim=-1)

    model_kl = kl_divergence(log_ngrams, logprobs)
    model.train()
    return model_kl


def load_tokens_and_ngrams(data_fp: Path, n: int):
    tokens = torch.from_numpy(
        np.load(data_fp / 'tokens.npy')
    )
    ngram_counts = torch.from_numpy(
        sparse.load_npz(data_fp / f'{n}.npz').todense()
    )
    return tokens, ngram_counts


def main(args):
    api = HfApi()
    ngram_values = range(args.min_n, args.max_n + 1)
    model_revisions = get_hf_revisions(api, args.model_path)
    model_cache_dir = Path("./pythia-70m-deduped/")
    ngram_dir = Path('/mnt/ssd-1/dashiell/ngram_data/1')
    write_dir = Path('/mnt/ssd-1/dashiell/pythia_ngrams/70m_standard')
    batch = torch.from_numpy(np.load(ngram_dir / 'tokens.npy'))

    for n in ngram_values:
        all_ngram_data = []
        ngram_counts = torch.from_numpy(np.load(ngram_dir / f'{n}.npy'))
        for revision in tqdm(model_revisions):
            num_steps = int(revision.lstrip('step'))
            model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-70m-deduped",
            revision=revision,
            cache_dir=model_cache_dir / revision,
            )
            model = model.to('cuda')
            kl_divs = estimate_model_kl(model, n, batch, ngram_counts)
            all_ngram_data.append(kl_data_to_df(kl_divs, num_steps, n))
        data = pl.concat(all_ngram_data)
        data.write_parquet(write_dir / f'{n}.parquet')
                                       

if __name__ == '__main__':
    args, _ = parser.parse_known_args()
    main(args)
