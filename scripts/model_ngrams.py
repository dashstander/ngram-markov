import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm

from ngram_markov.hooked_transformer import HookedTransformer
from ngram_markov.utils import nanogpt_to_hooked_transformer_config, convert_nanogpt_weights


def load_tl_model(path):
    ckpt = torch.load(path, map_location='cpu')
    config = nanogpt_to_hooked_transformer_config(ckpt['model_args'])
    tl_weights = convert_nanogpt_weights(ckpt['model'], config)
    tl_model = HookedTransformer(config)
    tl_model.load_state_dict(tl_weights)
    return tl_model


def ngram_iterator(n: int, num_tokens: int, batch_size: int, device):
    ngrams = torch.cartesian_prod(*[torch.arange(num_tokens) for _ in range(n)])
    for i, batch in enumerate(ngrams.split(batch_size)):
        yield i, batch.to(device, non_blocking=True)


@torch.no_grad()
def calculate_model_ngrams(model, data):
    return model(data)[:, -1, :].to('cpu').numpy()


if __name__ == '__main__':
    epoch = 53_000
    n = 3
    num_tokens = 512
    batch_size = 2 ** 17
    device = 'cuda:0'
    model_path = Path('/media/External01/ngram-checkpoints/4layer_tinystories')
    #output_path = Path('/media/External01/model-ngram-outputs')
    #output_path.mkdir(parents=True, exist_ok=True)
    model = load_tl_model(model_path / f'ckpt{epoch}.pt')
    model.eval()
    output_file = Path(f'ngram_{n}_outputs_epoch_{epoch}.npy')

    # Create a memory-mapped 3D numpy array
    mm_array = np.memmap(output_file, dtype='float32', mode='w+', shape=(num_tokens, num_tokens, num_tokens, num_tokens))

    # Calculate the number of batches
    total_ngrams = num_tokens ** n
    num_batches = (total_ngrams + batch_size - 1) // batch_size

    for i, ngrams in tqdm(ngram_iterator(n, num_tokens, batch_size, device), total=num_batches):
        outs = calculate_model_ngrams(model, ngrams)
        
        # Convert batch indices to 3D indices
        idx0 = ngrams[:, 0].cpu().numpy()
        idx1 = ngrams[:, 1].cpu().numpy()
        idx2 = ngrams[:, 2].cpu().numpy()
        
        # Write to the 3D array
        mm_array[idx0, idx1, idx2] = outs

        if i % 50 == 0:
            mm_array.flush()
    
    mm_array.flush()

    print(f"N-gram analysis complete. Output saved to {output_file}")

    # Verification step
    print("Verifying data...")
    mm_array_verify = np.memmap(output_file, dtype='float32', mode='r', shape=(num_tokens, num_tokens, num_tokens, num_tokens))
    print(f"Shape of saved array: {mm_array_verify.shape}")
    print(f"Sum of all elements: {np.sum(mm_array_verify)}")
    print(f"Mean of all elements: {np.mean(mm_array_verify)}")
    print(f"First few elements:\n{mm_array_verify[0, 0, 0, :5]}")
