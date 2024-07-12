from copy import deepcopy
from functools import cache
import numpy as np
import torch


def kl_divergence(log_p, log_q):
    return torch.sum(torch.exp(log_p) * (log_p - log_q), dim=-1)


def create_ngrams(tensor, n):
    assert n >= 1
    stride = tensor.stride()
    size = tensor.size()

    # Create a new stride that moves by 1 element in the original tensor
    new_stride = (stride[0], stride[1], stride[1])

    # Calculate the new shape of the tensor
    new_size = (size[0], size[1] - n + 1, n)

    # Use as_strided to create a new view of the tensor with the desired shape and stride
    ngrams = torch.as_strided(tensor, size=new_size, stride=new_stride)
    return ngrams


def calculate_ngram_kl_divergence(model, tokens, index, n):
    # Get the model's logits
    with torch.no_grad():
        logits, _ = model(tokens)

    if n == 1:
        # Get the unigram distributions from the 1-D vector
        ngrams = np.load('data/tinystories/ngrams/1grams.npy')
        unigram_distribution = torch.tensor(ngrams, device=tokens.device)[None, None, :]
        unigram_distribution /= unigram_distribution.sum()
        log_ngram_probs = torch.log(unigram_distribution + 1e-10)  # Add a small epsilon to avoid log(0)
        log_model_probs = torch.log_softmax(logits, dim=-1)
    else:
        token_ngrams = create_ngrams(tokens, n-1)
        vocab_size = 512
        ngram_counts = torch.tensor(
            index.batch_count_next(
                token_ngrams.reshape(-1, n-1).cpu().numpy(),
                511
            ),
            dtype=torch.float32,
            device=tokens.device
        )
        ngram_distributions = ngram_counts / ngram_counts.sum(axis=1)[:, None]
        ngram_distributions = ngram_distributions.view(tokens.shape[0], tokens.shape[1] - n + 2, vocab_size)
        
        log_ngram_probs = torch.log(ngram_distributions + 1e-10)  # Add a small epsilon to avoid log(0)
        log_model_probs = torch.log_softmax(logits[:, n-2:].contiguous(), dim=-1)
        
    # Calculate the KL divergence between the n-gram distributions and the model's logits
    kl_div = kl_divergence(log_ngram_probs, log_model_probs)

    return kl_div


@cache
def get_unigram_probabilities(index, num_tokens):
    queries = [[i] for i in range(num_tokens)]
    bigram_counts = np.array(index.batch_count_next(queries, num_tokens - 1), dtype=np.float32)
    unigram_counts = bigram_counts.sum(axis=1)
    return unigram_counts / unigram_counts.sum()


def ngram_inclusion_exclusion(context, index, num_tokens=512):
    """
    Compute n-gram probabilities and apply inclusion-exclusion principle.
    
    Args:
    context (list): The context sequence.
    index: The index object for counting n-grams.
    num_tokens (int): The total number of possible tokens.
    
    Returns:
    numpy.ndarray: A 2D array where each row represents the contribution
                   from that level of context.
    """
    function_values = np.zeros((len(context) + 1, num_tokens), dtype=np.float32)
    
    # Compute unigram probabilities
    unigram_probabilities = get_unigram_probabilities(index, num_tokens)
    function_values[0, :] = unigram_probabilities
    
    # Compute n-gram probabilities
    queries = [context[i:] for i in reversed(range(len(context)))]
    probs = np.array(index.batch_count_next(queries, num_tokens - 1), dtype=np.float32)
    probs /= probs.sum(axis=1)[:, None]
    
    # Apply inclusion-exclusion principle
    for k in range(1, len(context) + 1):
        function_values[k, :] = probs[k-1, :] - function_values[k-1, :]
    
    return function_values, probs