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


def calculate_ngram_kl_divergence(model, tokens, ngrams, n):
    # Get the model's logits
    with torch.no_grad():
        logits, _ = model(tokens)

    if n == 1:
        # Get the unigram distributions from the 1-D vector
        unigram_distribution = torch.tensor(ngrams, device=tokens.device)[None, None, :]
        unigram_distribution /= unigram_distribution.sum()
        log_ngram_probs = torch.log(unigram_distribution + 1e-10)  # Add a small epsilon to avoid log(0)
        log_model_probs = torch.log_softmax(logits, dim=-1)
    else:
        # Get the n-grams from the tokens
        ngrams_tensor = create_ngrams(tokens, n-1)

        # Convert n-grams to indices for accessing the sparse matrix
        vocab_size = ngrams.shape[1]
        exponents = vocab_size **  torch.flip(torch.arange(n - 1, device=tokens.device).view(1, 1, -1), [-1])
        indices = torch.sum(ngrams_tensor * exponents, dim=-1).detach().cpu()

        # Get the n-gram distributions from the sparse matrix
        ngram_counts = torch.tensor(ngrams[indices.view(-1).numpy()].toarray(), device=tokens.device)
        ngram_distributions = ngram_counts / ngram_counts.sum(axis=1)[:, None]
        ngram_distributions = ngram_distributions.view(tokens.shape[0], tokens.shape[1] - n + 2, vocab_size)

        log_ngram_probs = torch.log(ngram_distributions + 1e-10)  # Add a small epsilon to avoid log(0)
        log_model_probs = torch.log_softmax(logits[:, n-2:].contiguous(), dim=-1)
        
    # Calculate the KL divergence between the n-gram distributions and the model's logits
    kl_div = kl_divergence(log_ngram_probs, log_model_probs)

    return kl_div