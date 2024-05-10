from functools import partial
from collections import deque
import numpy as np


def vocab_to_idx_generator(n_vocab, *indices):
    base_sixteen = reversed([n_vocab**i for i in range(len(indices))])
    return sum([b * idx for (b, idx) in zip(base_sixteen, indices)])


def project_probabilities(probs, unigram_probs):
    """Uses the algorithm of alternating between making the rows stochastic (sum to 1.) and columns sum to `unigram_probs`, an adaptation of the algorithm described here to generate random bistochastic matrices: https://iopscience.iop.org/article/10.1088/1751-8113/42/36/365209. Does not end up sampling from the Haar measure on bistochastic matrices, but it converges very quickly.
    """
    curr_sums = probs.sum(axis=0)
    temp_probs = probs * (unigram_probs / curr_sums)[None, :]
    temp_sums = temp_probs.sum(axis=1)[:, None]
    return temp_probs / temp_sums


class NgramSampler:
    
    zipf_epsilon = 1.e-5
    zipf_max_iter = 20

    """Samples an n-gram distribution over a vocabulary of size `num_vocab` and degree `n`. The transition probabilities are given by an (num_vocab**(n-1), num_vocab) matrix where all of the rows sum to 1. The distribution of the next token depends only on the history of the prior n-1 tokens. This class will also sample an n-gram distribution consistent at the unigram level with a given Zipfian distribution. 
    """


    def __init__(self, num_vocab: int, n: int, transition_probs, rng = None):
        self.n = n
        self.num_vocab = num_vocab
        self.num_states = num_vocab ** (n - 1)
        self.transition_probs = transition_probs
        self.unigram_probs = transition_probs.sum(axis=0) / self.num_states
        self.rng = rng if rng is not None else np.random.default_rng()
        self.vocab_idx_fn = partial(vocab_to_idx_generator, num_vocab)
        assert transition_probs.shape == (self.num_states, self.num_vocab), (self.num_states, self.num_vocab)

        self._state = deque()

    @classmethod
    def sample_zipfian_distribution(cls, num_vocab: int, n: int, zipf_alpha: float, rng = None):
        if rng is None:
            rng = np.random.default_rng()
        num_states = num_vocab ** (n - 1)
        transition_probs = rng.dirichlet(np.ones((num_vocab,)), size=(num_states,))
        zipf_probs = (np.arange(num_vocab) + 1) ** (-1 * zipf_alpha)
        zipf_probs /= zipf_probs.sum()
        curr_unigram_probs = transition_probs.sum(axis=0) / num_states
        i = 0
        while not np.allclose(curr_unigram_probs, zipf_probs, atol=cls.zipf_epsilon):
            if i >= cls.zipf_max_iter:
                raise ValueError('Distribution did not converge to Zipfian on unigrams')
            transition_probs = project_probabilities(transition_probs, zipf_probs)
            curr_unigram_probs = transition_probs.sum(axis=0) / num_states
            i += 1
        return cls(num_vocab, n, transition_probs)
    
    def clear_state(self):
        self._state.clear()

    def sample(self):
        if len(self._state) < (self.n - 1):
            a = self.rng.choice(self.num_states, p=self.unigram_probs)
            self._state.append(a)
            return a
        else:
            idx = self.vocab_idx_fn(*tuple(self._state))
            a = self.rng.choice(self.num_vocab, p=self.transition_probs[idx])
            self._state.append(a)
            self._state.popleft()
            return a
